from torch import nn, optim

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

__all__ = [
    "hyper_parameters",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


def hyper_parameters() -> HyperParameters:
    r"""Hyper parameters."""
    return HyperParameters(
        content_loss=HyperParameters(layer="relu4_2", score_weight=1e0,),
        gram_style_loss=HyperParameters(
            layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
            layer_weights="mean",
            score_weight=5e0,
        ),
        mrf_style_loss=HyperParameters(
            layers=("relu3_1", "relu4_1"),
            layer_weights="mean",
            patch_size=3,
            stride=2,
            score_weight=1e-4,
        ),
        content_transform=HyperParameters(image_size=368, edge="short"),
        style_transform=HyperParameters(edge_size=368, edge="long"),
        batch_sampler=HyperParameters(num_batches=1000, batch_size=4),
    )


def preprocessor() -> nn.Module:
    return transforms.CaffePreprocessing()


def postprocessor() -> nn.Module:
    return transforms.CaffePostprocessing()


def multi_layer_encoder() -> enc.VGGMultiLayerEncoder:
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=True, allow_inplace=True
    )


def optimizer(transformer: nn.Module) -> optim.Adam:
    return optim.Adam(transformer.parameters(), lr=1e-3)
