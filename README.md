# ML Project Template

This is a starter template for machine learning projects in PyTorch.

The core of this library lives over [here](https://github.com/codekansas/ml-starter).

## Run a command

Train a ResNet18 model on CIFAR10:

```bash
runml train configs/image_demo.yaml
```

Train an RL PPO model on BipedalWalker:

```bash
runml train configs/rl_demo.yaml
```

Launch a Slurm job (requires setting the `SLURM_PARTITION` environment variable):

```bash
runml mp_train configs/image_demo.yaml trainer.name=slurm_sl
```

## Architecture

A new project is broken down into five parts:

1. _Task_: Defines the dataset and calls the model on a sample. Similar to a [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).
2. _Model_: Just a PyTorch `nn.Module`
3. _Trainer_: Defines the main training loop, and optionally how to distribute training when using multiple GPUs
4. _Optimizer_: Just a PyTorch `optim.Optimizer`
5. _LR Scheduler_: Just a PyTorch `optim.LRScheduler`

Most projects should just have to implement the Task and Model, and use a default trainer, optimizer and learning rate scheduler. Running the training command above will log the location of each component.

New tasks, models, trainers, optimizers and learning rate schedulers are added using the same API, although each should implement different things. For example, to create a new model, make a new file under `ml/models` and add the following code:

```python
from dataclasses import dataclass

from ml.core.config import conf_field
from ml.core.registry import register_model
from ml.models.base import BaseModel, BaseModelConfig


@dataclass
class NewModelConfig(BaseModelConfig):
  some_param: int = conf_field(10)


@register_model("new_model", NewModelConfig)
class NewModel(BaseModel[NewModelConfig]):
  def forward(self, x):
    return x + self.config.some_param
```

The framework will automatically search in all of the files in `ml/models` to populate the model registry. In your config file, you can then reference the registered model using whatever key you chose:

```yaml
model:
  name: new_model
```

Similar APIs exist for tasks, trainers, optimizers and learning rate schedulers. Try running the demo config to get a sense for how each of these fit together.

## Features

This repository implements some features which I find useful when starting ML projects.

### C++ Extensions

This template makes it easy to add custom C++ extensions to your PyTorch project. The demo includes a custom TorchScript-compatible nucleus sampling function, although more complex extensions are possible.

- [Custom TorchScript Op Tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
- [PyTorch CMake Extension Reference](https://github.com/pytorch/extension-script)

### Github Actions

This template automatically runs `black`, `isort`, `pylint` and `mypy` against your repository as a Github action. You can enable push-blocking until these tests pass.

### Lots of Timers

The training loop is pretty well optimized, but sometimes you can do stupid things when implementing a task that impact your performance. This adds a lot of timers which make it easy to spot likely training slowdowns, or you can run the full profiler if you want a more detailed breakdown.

### Compiled models

By default, models are run using `torch.compile`. To disable this behavior and use eager mode execution, set `TORCH_COMPILE=0`. If you try to launch a Slurm job with this flag set, it will show a warning.
