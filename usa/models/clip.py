# pylint: disable=too-many-lines
"""Defines handlers for OpenAI's pretrained CLIP model.

Most of this file is taken from here:
    https://github.com/openai/CLIP/blob/main/clip/model.py
"""

import functools
import gzip
import html
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, cast, get_args, overload

import ftfy
import ml.api as ml
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image as PILImage
from pkg_resources import packaging
from torch import Tensor, nn
from torchvision.datasets.utils import download_url

URL_PREFIX = "https://openaipublic.azureedge.net/clip/models"

PretrainedModel = Literal[
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT_B_32",
    "ViT_B_16",
    "ViT_L_14",
    "ViT_L_14_336px",
]


def cast_pretrained_model_key(s: str) -> PretrainedModel:
    args = get_args(PretrainedModel)
    assert s in args, f"Invalid pretraiend model key: '{s}' Valid options are {args}"
    return cast(PretrainedModel, s)


PRETRAINED_MODELS: dict[PretrainedModel, str] = {
    "RN50": f"{URL_PREFIX}/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": f"{URL_PREFIX}/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": f"{URL_PREFIX}/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": f"{URL_PREFIX}/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": f"{URL_PREFIX}/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT_B_32": f"{URL_PREFIX}/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT_B_16": f"{URL_PREFIX}/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT_L_14": f"{URL_PREFIX}/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT_L_14_336px": f"{URL_PREFIX}/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",  # noqa, pylint: disable=line-too-long
}

CLIP_VOCABULARY = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"


def _convert_image_to_rgb(image: PILImage) -> PILImage:
    return image.convert("RGB")


def preprocess(n_px: int) -> torchvision.transforms.Compose:
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(n_px, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ],
    )


@functools.lru_cache()
def default_bpe() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@functools.lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings. This means you need a
    large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around
    5K for decent coverage. This is a signficant percentage of your normal,
    say, 32K bpe vocab. To avoid that, we want lookup tables between utf-8
    bytes and unicode strings. And avoids mapping to whitespace/control
    characters the BPE code barfs on.

    Returns:
        Mapping from UTF-8 byte to unicode string.
    """

    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    css = [chr(n) for n in cs]
    return dict(zip(bs, css))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class CLIPTokenizer:
    def __init__(self) -> None:
        vocab_file_name = "CLIP_vocabulary.txt.gz"
        bpe_path = ml.get_model_dir() / vocab_file_name
        if not bpe_path.exists():
            download_url(CLIP_VOCABULARY, ml.get_model_dir(), filename=vocab_file_name)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges_unzipped = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges_unzipped = merges_unzipped[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges_unzipped]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\w+|\d|[^\s\w\d]+",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word_list: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word_list.extend(word[i:j])
                    i = j
                except Exception:
                    new_word_list.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word_list.append(first + second)
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens: list[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

    def tokenize(
        self,
        texts: str | list[str],
        context_length: int = 77,
        truncate: bool = False,
    ) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False, device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm2d(planes, device=device, dtype=dtype)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(planes, device=device, dtype=dtype)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, device=device, dtype=dtype)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # Downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            pool = nn.AvgPool2d(stride)
            conv = nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False, device=device, dtype=dtype)
            bn = nn.BatchNorm2d(planes * self.expansion, device=device, dtype=dtype)
            self.downsample = nn.Sequential(OrderedDict([("-1", pool), ("0", conv), ("1", bn)]))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if dtype is None:
            pos_emb = torch.randn(spacial_dim**2 + 1, embed_dim, device=device) / embed_dim**0.5
        else:
            pos_emb = torch.randn(spacial_dim**2 + 1, embed_dim, device=device, dtype=dtype) / embed_dim**0.5
        self.positional_embedding = nn.Parameter(pos_emb)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, device=device, dtype=dtype)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    __constants__ = ["input_resolution", "output_dim"]

    def __init__(
        self,
        layers: tuple[int, int, int, int],
        output_dim: int,
        heads: int,
        input_resolution: int = 224,
        width: int = 64,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """ResNet class that is similar to TorchVision's but with some changes.

        - There are now 3 "stem" convolutions as opposed to 1, with an average
          pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is
          prepended to convolutions with stride > 1
        - The final pooling layer is a QKV attention instead of an average pool

        Args:
            layers: Layer counts for the four parts of the ResNet
            output_dim: Number of final output dimensions
            heads: Number of attention heads
            input_resolution: Number of pixels in width and height directions
            width: Hidden channel count
            device: Default PyTorch device to use
            dtype: Default PyTorch dtype to use
        """

        super().__init__()

        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # The 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.bn1 = nn.BatchNorm2d(width // 2, device=device, dtype=dtype)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.bn2 = nn.BatchNorm2d(width // 2, device=device, dtype=dtype)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False, device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm2d(width, device=device, dtype=dtype)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # Residual layers
        self._inplanes = width  # This is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # The ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32,
            embed_dim,
            heads,
            output_dim,
            device=device,
            dtype=dtype,
        )

    def initialize_parameters(self) -> None:
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def stem(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class QuickGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, device=device, dtype=dtype)
        self.ln_1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4, device=device, dtype=dtype)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model, device=device, dtype=dtype)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor) -> Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: Tensor | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        blocks = [ResidualAttentionBlock(width, heads, attn_mask, device=device, dtype=dtype) for _ in range(layers)]
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    __constants__ = ["input_resolution", "output_dim"]

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        scale = width**-0.5

        if dtype is None:
            class_emb = scale * torch.randn(width, device=device)
            pos_emb = scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width, device=device)
            proj = scale * torch.randn(width, output_dim, device=device)
        else:
            class_emb = scale * torch.randn(width, device=device, dtype=dtype)
            pos_emb = scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width, device=device, dtype=dtype)
            proj = scale * torch.randn(width, output_dim, device=device, dtype=dtype)

        self.class_embedding = nn.Parameter(class_emb)
        self.positional_embedding = nn.Parameter(pos_emb)
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.transformer = Transformer(width, layers, heads, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.proj = nn.Parameter(proj)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            device=device,
            dtype=dtype,
        )

        if dtype is None:
            pos_emb = torch.empty(self.context_length, transformer_width, device=device)
            text_proj = torch.empty(transformer_width, embed_dim, device=device)
        else:
            pos_emb = torch.empty(self.context_length, transformer_width, device=device, dtype=dtype)
            text_proj = torch.empty(transformer_width, embed_dim, device=device, dtype=dtype)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width, device=device, dtype=dtype)
        self.positional_embedding = nn.Parameter(pos_emb)
        self.ln_final = nn.LayerNorm(transformer_width, device=device, dtype=dtype)
        self.text_projection = nn.Parameter(text_proj)

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self) -> Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: Tensor) -> Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: tuple[int, int, int, int] | int,
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.visual: ModifiedResNet | VisionTransformer
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                device=device,
                dtype=dtype,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                device=device,
                dtype=dtype,
            )

        self.linguistic = TextModel(
            embed_dim=embed_dim,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            device=device,
            dtype=dtype,
        )

        if dtype is None:
            logit_scale = torch.ones([], device=device) * np.log(1 / 0.07)
        else:
            logit_scale = torch.ones([], device=device, dtype=dtype) * np.log(1 / 0.07)

        self.logit_scale = nn.Parameter(logit_scale)

        self.initialize_parameters()

    @torch.jit.ignore
    def get_preprocess(self) -> torchvision.transforms.Compose:
        return preprocess(self.visual.input_resolution)

    def initialize_parameters(self) -> None:
        if isinstance(self.visual, ModifiedResNet):
            self.visual.initialize_parameters()
        self.linguistic.initialize_parameters()

    def build_attention_mask(self) -> Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image: Tensor) -> Tensor:
        return self.visual(image)

    def encode_text(self, text: Tensor) -> Tensor:
        return self.linguistic(text)

    def forward(self, image: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module) -> None:
    """Convert applicable model parameters to fp16.

    Args:
        model: The model to convert
    """

    def _convert_weights_to_fp16(mod: nn.Module) -> None:
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            mod.weight.data = mod.weight.data.half()
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.half()

        if isinstance(mod, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(mod, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(mod, name):
                attr = getattr(mod, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@overload
def load_pretrained(
    key: PretrainedModel | nn.Module,
    mode: Literal["visual"],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ModifiedResNet | VisionTransformer:
    ...


@overload
def load_pretrained(
    key: PretrainedModel | nn.Module,
    mode: Literal["linguistic"],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TextModel:
    ...


@overload
def load_pretrained(
    key: PretrainedModel | nn.Module,
    mode: Literal["all"],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> CLIP:
    ...


def load_pretrained(
    key: PretrainedModel | nn.Module,
    mode: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> CLIP | ModifiedResNet | VisionTransformer | TextModel:
    """Builds the CLIP model from a state dictionary.

    Args:
        key: The model key to load, or another model to load weights from
        mode: Default is to return all models, but can optionally return just
            the visual or linguistic part of the model
        device: The device for the model
        dtype: The dtype for the model

    Returns:
        The constructed clip model, or just the visual or text branch
    """

    assert mode in ("all", "visual", "linguistic")

    if isinstance(key, nn.Module):
        ckpt = key.state_dict()
    else:
        filepath = get_pretrained_path(key)
        ckpt = torch.jit.load(filepath, map_location="cpu").state_dict()

    vit = "visual.proj" in ckpt

    vision_layers: tuple[int, int, int, int] | int
    if vit:
        vision_width = ckpt["visual.conv1.weight"].shape[0]
        vision_layers = sum(1 for k in ckpt.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight"))
        vision_patch_size = ckpt["visual.conv1.weight"].shape[-1]
        grid_size = round((ckpt["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        vision_layers = cast(
            tuple[int, int, int, int],
            tuple(len(set(k.split(".")[2] for k in ckpt if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]),
        )
        vision_width = ckpt["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((ckpt["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == ckpt["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = ckpt["text_projection"].shape[1]
    context_length = ckpt["positional_embedding"].shape[0]
    vocab_size = ckpt["token_embedding.weight"].shape[0]
    transformer_width = ckpt["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in ckpt if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        device=device,
        dtype=dtype,
    )

    for ckpt_key in ["input_resolution", "context_length", "vocab_size"]:
        if ckpt_key in ckpt:
            del ckpt[ckpt_key]

    # Prepends `linguistic.` prefix to linguistic weights.
    non_visual_keys = {k: v for k, v in ckpt.items() if not k.startswith("visual.")}
    non_visual_keys.pop("logit_scale")
    for k, v in non_visual_keys.items():
        del ckpt[k]
        ckpt[f"linguistic.{k}"] = v

    convert_weights(model)

    def get_ckpt_part(ckpt: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}

    if mode == "visual":
        model.visual.load_state_dict(get_ckpt_part(ckpt, "visual."))
        return model.visual
    elif mode == "linguistic":
        model.linguistic.load_state_dict(get_ckpt_part(ckpt, "linguistic."))
        return model.linguistic
    else:
        model.load_state_dict(ckpt)
        return model


def get_pretrained_path(key: PretrainedModel) -> Path:
    if key not in PRETRAINED_MODELS:
        raise KeyError(f"Invalid CLIP model key {key}; choices are {list(PRETRAINED_MODELS.keys())}")
    model_url = PRETRAINED_MODELS[key]
    save_path = (ml.get_model_dir() / f"CLIP_{key}").resolve()
    filename = "ckpt.pt"
    filepath = save_path / filename
    if not filepath.exists():
        download_url(model_url, str(save_path), filename=filename)
    return filepath


def test_pretrained_model(model_key: PretrainedModel) -> None:
    """Tests the pretrained model implementation against JIT'd version.

    This also provides a reference for how to call each model.

    Usage:
        python -m ml.models.clip

    Args:
        model_key: The pretrained model key
    """

    # Gets an image of a peach from Wikipedia.
    peach_url = "https://upload.wikimedia.org/wikipedia/commons/9/9e/Autumn_Red_peaches.jpg"
    url_path = Path("/tmp/peach.jpg")
    if not url_path.exists():
        download_url(peach_url, "/tmp", filename="peach.jpg")

    peach_img = PILImage.open(url_path)
    pos_desc = "A picture of an Autumn Red peach"
    neg_desc = "An Instagram photo of a cute puppy"

    # Loads the JIT'd model and the regular model.
    auto_device = ml.AutoDevice.detect_device()
    device, dtype = auto_device.get_device(), torch.half
    jit_model = cast(CLIP, torch.jit.load(get_pretrained_path(model_key), map_location="cpu"))
    model = load_pretrained(jit_model, "all")

    # Moves to the correct device.
    jit_model = jit_model.to(device).eval()
    model = model.to(device, dtype).eval()

    # Converts raw inputs to tensors.
    img_tensorizer = model.get_preprocess()
    tokenizer = CLIPTokenizer()
    imgs = img_tensorizer(peach_img).to(device, dtype).unsqueeze(0).repeat_interleave(2, dim=0)
    texts = tokenizer.tokenize([pos_desc, neg_desc]).to(device)

    with torch.no_grad():
        img_out = model.encode_image(imgs)
        text_out = model.encode_text(texts)
        img_ref = jit_model.encode_image(imgs)
        text_ref = jit_model.encode_text(texts)

    # Checks the tensors against each other.
    assert ((img_out - img_ref).abs().mean() / img_ref.abs().mean()).item() < 5e-2
    assert ((text_out - text_ref).abs().mean() / text_ref.abs().mean()).item() < 5e-2

    with torch.no_grad():
        img_out_preds, text_out_preds = model(imgs, texts)
        img_ref_preds, text_ref_preds = jit_model(imgs, texts)

    # Checks that the positive description scores more highly than the negative description.
    assert (img_ref_preds[:, 0] > img_ref_preds[:, 1]).all().item()

    # Checks model against JIT'd model.
    assert ((img_out_preds - img_ref_preds).abs().max() / img_ref_preds.abs().mean()).item() < 5e-2
    assert ((text_out_preds - text_ref_preds).abs().max() / text_ref_preds.abs().mean()).item() < 5e-2


if __name__ == "__main__":
    test_pretrained_model("RN50")
