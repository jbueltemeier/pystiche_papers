from os import path
from typing import Dict, List, Optional, Sequence, Tuple, cast, Union

from torch import nn

from pystiche import core
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

VGG_DATA: Dict[str, Dict[str, Union[int, str, List[Union[str, int]]]]] = {
    "relu5_1": {"filename": "feature_invertor_conv5_1.pth",
                "in_channels": 512,
                "cfgs": ["R", 512, "U", "R", 512, "R", 512, "R", 512, "R", 256, "U", "R", 256, "R", 256, "R", 256, "R", 128, "U", "R", 128, "R", 64, "U", "R", 64, "R"],
                },
    "relu4_1": {"filename": "feature_invertor_conv4_1.pth",
                "in_channels": 512,
                "cfgs": ["R", 256, "U", "R", 256, "R", 256, "R", 256, "R", 128, "U", "R", 128, "R", 64, "U", "R", 64, "R"],
                },
    "relu3_1": {"filename": "feature_invertor_conv3_1.pth",
                "in_channels": 256,
                "cfgs": ["R", 128, "U", "R", 128, "R", 64, "U", "R", 64, "R"],
                },
    "relu2_1": {"filename": "feature_invertor_conv2_1.pth",
                "in_channels": 128,
                "cfgs": ["R", 64, "U", "R", 64, "R"],
                },
    "relu1_1": {"filename": "feature_invertor_conv1_1.pth",
                "in_channels": 64,
                "cfgs": ["R",],
                },
}



class SequentialDecoder(core.SequentialModule):
    r"""Decoder that operates in sequential manner.

    Args:
        modules: Sequential modules.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(*modules)


class VGGDecoderBuilder(object):
    def __init__(self) -> None:
        super().__init__()

    def conv_block(self, in_channels: int, out_channels: int) -> Sequence[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU()
        ]


    def output_conv(self) -> nn.Module:
        return nn.Conv2d(64, 3, kernel_size=3)

    def build_model(self, in_channels:int, cfgs: List[Union[str, int]]) -> SequentialDecoder:
        modules: List[nn.Module] = []
        for cfg in cfgs:
            if isinstance(cfg, int):
                modules.extend(self.conv_block(in_channels, cfg))
                in_channels = cfg
            elif cfg == "R":
                modules.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            else:  # cfg == "U"
                modules.append(nn.UpsamplingNearest2d(scale_factor=2))

        modules.append(self.output_conv())

        return SequentialDecoder(modules)


class VGGDecoderLoader(ModelLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root=root)

    def build_model(self, name: str) -> None:  # type: ignore[override]
        builder = VGGDecoderBuilder()
        data = VGG_DATA[name]
        self.models[name] = builder.build_model(data["in_channels"], data["cfgs"])

    def load_models(
        self, init_weights: bool = True, layers: Optional[Sequence[str]] = None
    ) -> Dict[str, core.SequentialModule]:
        if layers is None:
            layers = VGG_DATA.keys()

        for layer in layers:
            self.build_model(layer)
            if init_weights:
                self.init_model(
                    cast(str, VGG_DATA[layer]["filename"]), layer)
        return self.models


class DecoderVGGModels(PretrainedVGGModels):
    def download_models(self) -> None:
        for id, filename in enumerate(DECODER_FILES, 1):
            self.download(id, filename)

    def load_models(self) -> Dict[str, core.SequentialModule]:
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
