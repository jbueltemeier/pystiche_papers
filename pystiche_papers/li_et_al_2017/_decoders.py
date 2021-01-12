from os import path
from typing import Dict, List, Optional, Sequence, Tuple, cast

from torch import nn

from pystiche import enc
from pystiche_papers.utils import HyperParameters

from ._utils import ModelLoader, PretrainedVGGModels, channel_progression
from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["SequentialDecoder", "VGGDecoderLoader", "vgg_decoders"]

DECODER_FILES = (
    "feature_invertor_conv1_1.pth",
    "feature_invertor_conv2_1.pth",
    "feature_invertor_conv3_1.pth",
    "feature_invertor_conv4_1.pth",
    "feature_invertor_conv5_1.pth",
)

VGG_DECODER_DATA = {
    1: {
        "name": "relu1_1",
        "channels": (64, 64, 3),
        "filename": "feature_invertor_conv1_1.pth",
    },
    2: {
        "name": "relu2_1",
        "channels": (128, 128, 64),
        "filename": "feature_invertor_conv2_1.pth",
    },
    3: {
        "name": "relu3_1",
        "channels": (256, 256, 256, 256, 128),
        "filename": "feature_invertor_conv3_1.pth",
    },
    4: {
        "name": "relu4_1",
        "channels": (512, 512, 512, 512, 256),
        "filename": "feature_invertor_conv4_1.pth",
    },
    5: {
        "name": "relu5_1",
        "channels": (512, 512),
        "filename": "feature_invertor_conv5_1.pth",
    },
}


class SequentialDecoder(enc.SequentialEncoder):
    r"""Decoder that operates in sequential manner.

    Args:
        modules: Sequential modules.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(modules=modules)


class VGGDecoderBuilder(object):
    def __init__(self) -> None:
        super().__init__()

    def conv_block(self, channels: Tuple[int]) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        channel_progression(
            lambda in_channels, out_channels: modules.extend(
                [
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3),
                    nn.ReLU(),
                ]
            ),
            channels=channels,
        )
        return modules

    def input_conv(self, layer: int) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        depth_data = VGG_DECODER_DATA[layer]
        modules.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        modules.append(
            nn.Conv2d(
                cast(Tuple, depth_data["channels"])[-2],
                cast(Tuple, depth_data["channels"])[-1],
                kernel_size=3,
            )
        )
        modules.append(nn.ReLU())
        return modules

    def output_conv(self) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        depth_data = VGG_DECODER_DATA[1]
        modules.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        modules.append(
            nn.Conv2d(
                cast(Tuple, depth_data["channels"])[-2],
                cast(Tuple, depth_data["channels"])[-1],
                kernel_size=3,
            )
        )
        return modules

    def depth_level(self, channels: Tuple[int]) -> Sequence[nn.Module]:
        modules: List[nn.Module] = [nn.UpsamplingNearest2d(scale_factor=2)]
        modules.extend(self.conv_block(channels))
        return modules

    def build_model(self, layer: int) -> SequentialDecoder:
        modules: List[nn.Module] = []

        if layer > 1:
            modules.extend(self.input_conv(layer))

            for depth in range(layer - 1, 1, -1):
                depth_data = VGG_DECODER_DATA[depth]
                modules.extend(
                    self.depth_level(cast(Tuple[int], depth_data["channels"]))
                )

            channels = cast(Tuple, VGG_DECODER_DATA[1]["channels"])[:-1]
            modules.extend(self.depth_level(cast(Tuple[int], channels)))

        modules.extend(self.output_conv())

        return SequentialDecoder(modules)


class VGGDecoderLoader(ModelLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root=root)

    def build_model(self, name: str, layer: int) -> None:  # type: ignore[override]
        builder = VGGDecoderBuilder()
        self.models[name] = builder.build_model(layer)

    def load_models(
        self, init_weights: bool = True, layers: Optional[Sequence[int]] = None
    ) -> Dict[str, enc.SequentialEncoder]:
        if layers is None:
            layers = cast(Sequence[int], VGG_DECODER_DATA.keys())

        for layer in layers:
            vgg_data = VGG_DECODER_DATA[layer]
            self.build_model(cast(str, vgg_data["name"]), layer)
            if init_weights:
                self.init_model(
                    cast(str, vgg_data["filename"]), cast(str, vgg_data["name"])
                )
        return self.models


class DecoderVGGModels(PretrainedVGGModels):
    def download_models(self) -> None:
        for id, filename in enumerate(DECODER_FILES, 1):
            self.download(id, filename)

    def load_models(self) -> Dict[str, enc.SequentialEncoder]:
        return cast(VGGDecoderLoader, self.loader).load_models(layers=self.layers)


def vgg_decoders(
    hyper_parameters: Optional[HyperParameters] = None,
) -> Dict[str, SequentialDecoder]:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    here = path.dirname(__file__)

    model_dir = path.join(here, "models")
    loader = VGGDecoderLoader(model_dir)
    vgg_decoder = DecoderVGGModels(
        model_dir, layers=hyper_parameters.decoder.layers, loader=loader
    )
    return cast(Dict[str, SequentialDecoder], vgg_decoder.load_models())
