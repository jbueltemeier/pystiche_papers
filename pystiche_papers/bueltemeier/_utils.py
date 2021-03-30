from typing import Optional, Sequence, Tuple

from torch import nn, optim

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.gatys_ecker_bethge_2016 import (
    compute_layer_weights as _compute_layer_weights,
)
from pystiche_papers.utils import HyperParameters

__all__ = [
    "preprocessor",
    "postprocessor",
    "optimizer",
    "multi_layer_encoder",
    "hyper_parameters",
]


def preprocessor() -> nn.Module:
    return transforms.CaffePreprocessing()


def postprocessor() -> nn.Module:
    return transforms.CaffePostprocessing()


def optimizer(transformer: nn.Module) -> optim.Adam:
    return optim.Adam(transformer.parameters(), lr=1e-3)


def multi_layer_encoder() -> enc.VGGMultiLayerEncoder:
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=True, allow_inplace=True
    )


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
    layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    return _compute_layer_weights(layers, multi_layer_encoder=multi_layer_encoder)


def hyper_parameters() -> HyperParameters:
    r"""Hyper parameters."""
    gram_style_loss_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    return HyperParameters(
        content_loss=HyperParameters(layer="relu4_2", score_weight=1e0,),
        gram_style_loss=HyperParameters(
            layers=gram_style_loss_layers,
            layer_weights=compute_layer_weights(gram_style_loss_layers),
            score_weight=1e2,
        ),
        mrf_style_loss=HyperParameters(
            layers=("relu3_1", "relu4_1"),
            layer_weights="mean",
            patch_size=3,
            stride=2,
            score_weight=1e-4,
        ),
        content_transform=HyperParameters(image_size=512, edge="short"),
        style_transform=HyperParameters(edge_size=512, edge="short"),
        batch_sampler=HyperParameters(num_batches=20000, batch_size=4),
    )
