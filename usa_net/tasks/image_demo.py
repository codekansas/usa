from dataclasses import dataclass

import ml.api as ml
import torch.nn.functional as F
import torchvision
from torch import Tensor

from usa_net.models.resnet import ResNetModel

CLASS_TO_IDX = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}


@dataclass
class ImageDemoTaskConfig(ml.SupervisedLearningTaskConfig):
    pass


@ml.register_task("image_demo", ImageDemoTaskConfig)
class ImageDemoTask(ml.SupervisedLearningTask[ImageDemoTaskConfig, ResNetModel, tuple[Tensor, Tensor], Tensor, Tensor]):
    def __init__(self, config: ImageDemoTaskConfig) -> None:
        super().__init__(config)

        # Gets the class names for each index.
        # class_to_idx = self.get_dataset("test").class_to_idx
        self.idx_to_class = {i: name for name, i in CLASS_TO_IDX.items()}

    def get_label(self, true_class: int, pred_class: int) -> str:
        return "\n".join(
            [
                f"True: {self.idx_to_class.get(true_class, 'MISSING')}",
                f"Predicted: {self.idx_to_class.get(pred_class, 'MISSING')}",
            ]
        )

    def run_model(self, model: ResNetModel, batch: tuple[Tensor, Tensor], state: ml.State) -> Tensor:
        image, _ = batch
        return model(image)

    def compute_loss(self, model: ResNetModel, batch: tuple[Tensor, Tensor], state: ml.State, output: Tensor) -> Tensor:
        (image, classes), preds = batch, output
        pred_classes = preds.argmax(dim=1, keepdim=True)

        # Passing in a callable function ensures that we don't compute the
        # metric unless it's going to be logged, for example, when the logger
        # is rate-limited.
        self.logger.log_scalar("accuracy", lambda: (classes == pred_classes).float().mean())

        # On validation and test steps, logs images to each image logger.
        if state.phase in ("valid", "test"):
            bsz = classes.shape[0]
            texts = [self.get_label(int(classes[i].item()), int(pred_classes[i].item())) for i in range(bsz)]
            self.logger.log_labeled_images("image", (image, texts))

        return F.cross_entropy(preds, classes.flatten().long(), reduction="none")

    def get_dataset(self, phase: ml.Phase) -> torchvision.datasets.CIFAR10:
        return torchvision.datasets.CIFAR10(
            root=ml.get_data_dir(),
            train=phase == "train",
            download=True,
        )
