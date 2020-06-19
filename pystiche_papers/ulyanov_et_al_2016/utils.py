from typing import Any, List

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from pystiche.enc import VGGMultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image.transforms.processing import CaffePostprocessing, CaffePreprocessing


def ulyanov_et_al_2016_multi_layer_encoder() -> VGGMultiLayerEncoder:
    return vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def ulyanov_et_al_2016_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def ulyanov_et_al_2016_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def ulyanov_et_al_2016_optimizer(
    transformer: nn.Module, impl_params: bool = True, instance_norm: bool = True
) -> optim.Adam:
    if impl_params:
        lr = 1e-3 if instance_norm else 1e-1
    else:
        lr = 1e-1
    return optim.Adam(transformer.parameters(), lr=lr)


class DelayedExponentialLR(ExponentialLR):
    last_epoch: int
    gamma: float
    base_lrs: List[float]

    def __init__(
        self, optimizer: Optimizer, gamma: float, delay: int, **kwargs: Any
    ) -> None:
        self.delay = delay
        super().__init__(optimizer, gamma, **kwargs)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        exp = self.last_epoch - self.delay + 1
        if exp > 0:
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def ulyanov_et_al_2016_lr_scheduler(
    optimizer: Optimizer, impl_params: bool = True,
) -> ExponentialLR:
    if impl_params:
        lr_scheduler = ExponentialLR(optimizer, 0.8)
    else:
        lr_scheduler = DelayedExponentialLR(optimizer, 0.7, 5)
    return lr_scheduler