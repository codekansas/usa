from dataclasses import dataclass

import ml.api as ml
from torch import Tensor


@dataclass
class TemplateModelConfig(ml.BaseModelConfig):
    pass


@ml.register_model("template", TemplateModelConfig)
class TemplateModel(ml.BaseModel[TemplateModelConfig]):
    def __init__(self, config: TemplateModelConfig) -> None:
        super().__init__(config)

        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
