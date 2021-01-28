from os import path
from typing import Dict, List, Optional, Sequence, Union, cast

from torch import nn

from pystiche import core
from pystiche_papers.utils import HyperParameters

from ._utils import ModelLoader, PretrainedVGGModels
from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["SequentialDecoder", "VGGDecoderBuilder", "VGGDecoderLoader", "vgg_decoders"]

DECODER_FILES = (
    "feature_invertor_conv1_1.pth",
    "feature_invertor_conv2_1.pth",
    "feature_invertor_conv3_1.pth",
    "feature_invertor_conv4_1.pth",
    "feature_invertor_conv5_1.pth",
)

VGG_DATA: Dict[Union[int, str], Dict[str, Union[int, str, List[Union[str, int]]]]] = {
    "relu5_1": {
        "filename": "feature_invertor_conv5_1.pth",
        "in_channels": 512,
        "cfgs": [
            "R",
            512,
            "U",
            "R",
            512,
            "R",
            512,
            "R",
            512,
            "R",
            256,
            "U",
            "R",
            256,
            "R",
            256,
            "R",
            256,
            "R",
            128,
            "U",
            "R",
            128,
            "R",
            64,
            "U",
            "R",
            64,
            "R",
        ],
    },
    "relu4_1": {
        "filename": "feature_invertor_conv4_1.pth",
        "in_channels": 512,
        "cfgs": [
            "R",
            256,
            "U",
            "R",
            256,
            "R",
            256,
            "R",
            256,
            "R",
            128,
            "U",
            "R",
            128,
            "R",
            64,
            "U",
            "R",
            64,
            "R",
        ],
    },
    "relu3_1": {
        "filename": "feature_invertor_conv3_1.pth",
        "in_channels": 256,
        "cfgs": ["R", 128, "U", "R", 128, "R", 64, "U", "R", 64, "R"],
    },
    "relu2_1": {
        "filename": "feature_invertor_conv2_1.pth",
        "in_channels": 128,
        "cfgs": ["R", 64, "U", "R", 64, "R"],
    },
    "relu1_1": {
        "filename": "feature_invertor_conv1_1.pth",
        "in_channels": 64,
        "cfgs": ["R"],
    },
}


class SequentialDecoder(core.SequentialModule):
    def __init__(self, *modules: nn.Module, layer: Union[int, str]):
        super().__init__(*modules)
        self.layer = layer


class VGGDecoderBuilder(object):
    def __init__(self) -> None:
        super().__init__()

    def conv_block(self, in_channels: int, out_channels: int) -> Sequence[nn.Module]:
        return [nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU()]

    def output_conv(self) -> nn.Module:
        return nn.Conv2d(64, 3, kernel_size=3)

    def build_model(
        self, layer: Union[int, str], in_channels: int, cfgs: List[Union[int, str]]
    ) -> SequentialDecoder:
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

        return SequentialDecoder(*modules, layer=layer)


class VGGDecoderLoader(ModelLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root=root)
        self.builder = VGGDecoderBuilder()

    def load_model(
        self, layer: Union[int, str], init_weights: bool = True
    ) -> SequentialDecoder:
        data = VGG_DATA[layer]
        in_channels = cast(int, data["in_channels"])
        cfgs = cast(List[Union[int, str]], data["cfgs"])
        filename = cast(str, data["filename"])
        model = self.builder.build_model(layer, in_channels, cfgs)
        if init_weights:
            model = self.init_model(model, filename)  # type: ignore[assignment]
        return model


class DecoderVGGModels(PretrainedVGGModels):
    def download_models(self) -> None:
        for id, filename in enumerate(DECODER_FILES, 1):
            self.download(id, filename)

    def load_models(self) -> Dict[Union[int, str], SequentialDecoder]:  # type: ignore[override]
        if self.layers is None:
            self.layers = cast(Sequence[Union[int, str]], VGG_DATA.keys())

        models: Dict[Union[int, str], SequentialDecoder] = {}
        for layer in self.layers:
            models[layer] = cast(VGGDecoderLoader, self.loader).load_model(layer)
        return models


def vgg_decoders(
    hyper_parameters: Optional[HyperParameters] = None,
) -> Dict[Union[int, str], SequentialDecoder]:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    here = path.dirname(__file__)

    model_dir = path.join(here, "models")
    loader = VGGDecoderLoader(model_dir)
    vgg_decoder = DecoderVGGModels(
        model_dir, layers=hyper_parameters.decoder.layers, loader=loader
    )
    return vgg_decoder.load_models()
