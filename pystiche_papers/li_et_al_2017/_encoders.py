from os import path
from typing import List, Optional, Sequence, Tuple, cast

from torch import nn

from pystiche import enc

from ._utils import ModelLoader, PretrainedVGGModels, channel_progression

__all__ = ["VGGEncoderLoader", "vgg_multi_layer_encoder"]

BASE_URL = (
    "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"
)

ENCODER_FILES = (
    "vgg_normalised_conv1_1.pth",
    "vgg_normalised_conv2_1.pth",
    "vgg_normalised_conv3_1.pth",
    "vgg_normalised_conv4_1.pth",
    "vgg_normalised_conv5_1.pth",
)

VGG_ENCODER_DATA = {
    0: {"name": "input_conv", "channels": (3, 3), "filename": ""},
    1: {
        "name": "relu1_1",
        "channels": (3, 64, 64),
        "filename": "vgg_normalised_conv1_1.pth",
    },
    2: {
        "name": "relu2_1",
        "channels": (64, 128, 128),
        "filename": "vgg_normalised_conv2_1.pth",
    },
    3: {
        "name": "relu3_1",
        "channels": (128, 256, 256, 256, 256),
        "filename": "vgg_normalised_conv3_1.pth",
    },
    4: {
        "name": "relu4_1",
        "channels": (256, 512, 512, 512, 512),
        "filename": "vgg_normalised_conv4_1.pth",
    },
    5: {
        "name": "relu5_1",
        "channels": (512, 512),
        "filename": "vgg_normalised_conv5_1.pth",
    },
}


class VGGEncoderBuilder(object):
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

    def input_conv(self) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        depth_data = VGG_ENCODER_DATA[0]
        modules.append(
            nn.Conv2d(
                cast(Tuple, depth_data["channels"])[0],
                cast(Tuple, depth_data["channels"])[1],
                kernel_size=1,
            )
        )
        return modules

    def output_conv(self, depth: int) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        depth_data = VGG_ENCODER_DATA[depth]
        modules.extend(self.conv_block(cast(Tuple[int], depth_data["channels"])[:2]))
        return modules

    def depth_level(self, channels: Tuple[int]) -> Sequence[nn.Module]:
        modules: List[nn.Module] = []
        modules.extend(self.conv_block(channels))
        modules.append(nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True))
        return modules

    def build_model(self, layer: int) -> enc.SequentialEncoder:
        modules: List[nn.Module] = []
        modules.extend(self.input_conv())

        for depth in range(1, layer):
            depth_data = VGG_ENCODER_DATA[depth]
            modules.extend(self.depth_level(cast(Tuple[int], depth_data["channels"])))

        modules.extend(self.output_conv(layer))

        return enc.SequentialEncoder(modules)


class VGGEncoderLoader(ModelLoader):
    def __init__(self, root: str,) -> None:
        super().__init__(root=root)

    def build_model(self, name: str, layer: int) -> None:  # type: ignore[override]
        builder = VGGEncoderBuilder()
        self.models[name] = builder.build_model(layer)

    def load_models(
        self, init_weights: bool = True, layer: Optional[int] = None,
    ) -> enc.MultiLayerEncoder:
        if layer is None:
            layer = len(VGG_ENCODER_DATA.keys()) - 1

        vgg_data = VGG_ENCODER_DATA[layer]
        self.build_model(cast(str, vgg_data["name"]), layer)
        if init_weights:
            self.init_model(
                cast(str, vgg_data["filename"]), cast(str, vgg_data["name"])
            )

        return self._multi_layer_encoder(self.models[cast(str, vgg_data["name"])])

    def _multi_layer_encoder(
        self, encoder: enc.SequentialEncoder
    ) -> enc.MultiLayerEncoder:
        modules = []
        block = 1
        depth = 0
        for module in encoder._modules.values():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}_{depth}"
                if depth == 0:
                    depth += 1
            elif isinstance(module, nn.ReflectionPad2d):
                name = f"pad{block}_{depth}"
            elif isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
                name = f"relu{block}_{depth}"
                # each ReLU layer increases the depth of the current block
                depth += 1
            else:  # isinstance(module, nn.MaxPool2d):
                name = f"pool{block}"
                # each pooling layer marks the end of the current block
                block += 1
                depth = 1

            modules.append((name, module))

        return enc.MultiLayerEncoder(modules)


class EncoderVGGModel(PretrainedVGGModels):
    def download_models(self) -> None:
        for id, filename in enumerate(ENCODER_FILES, 1):
            self.download(id, filename)

    def load_models(self) -> enc.MultiLayerEncoder:
        return cast(VGGEncoderLoader, self.loader).load_models()


def vgg_multi_layer_encoder() -> enc.MultiLayerEncoder:
    here = path.dirname(__file__)
    model_dir = path.join(here, "models")
    loader = VGGEncoderLoader(model_dir)
    vgg_encoder = EncoderVGGModel(model_dir, loader=loader)
    return vgg_encoder.load_models()
