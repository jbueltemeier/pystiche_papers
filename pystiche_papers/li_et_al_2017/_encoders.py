from pystiche import enc
from torch import nn
from typing import Sequence, Tuple, Optional, Dict, List
from os import path

from ._utils import ModelLoader, channel_progression, PretrainedVGGModels

__all__ = ["VGGEncoderLoader", "vgg_encoders"]

BASE_URL = "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"

ENCODER_FILES = (
    "vgg_normalised_conv1_1.pth",
    "vgg_normalised_conv2_1.pth",
    "vgg_normalised_conv3_1.pth",
    "vgg_normalised_conv4_1.pth",
    "vgg_normalised_conv5_1.pth"
)
VGG_ENCODER_DATA = {
        0: {
            "name": "input_norm",
            "first_conv": (3, 3),
            "channels": (),
            "filename": ""
        },
        1: {
            "name": "relu1_1",
            "first_conv": (3, 64),
            "channels": (64, 64),
            "filename": "vgg_normalised_conv1_1.pth"
        },
        2: {
            "name": "relu2_1",
            "first_conv": (64, 128),
            "channels": (128, 128),
            "filename": "vgg_normalised_conv2_1.pth"
        },
        3: {
            "name": "relu3_1",
            "first_conv": (128, 256),
            "channels": (256, 256, 256, 256),
            "filename": "vgg_normalised_conv3_1.pth"
        },
        4: {
            "name": "relu4_1",
            "first_conv": (256, 512),
            "channels": (512, 512, 512, 512),
            "filename": "vgg_normalised_conv4_1.pth"
        },
        5: {
            "name": "relu5_1",
            "first_conv": (512, 512),
            "channels": (),
            "filename": "vgg_normalised_conv5_1.pth"
        },
    }


class VGGEncoderLoader(ModelLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root=root)

    def conv_block(self, channels: Tuple[int]):
        modules = []
        channel_progression(
                lambda in_channels, out_channels: modules.extend(
                    [nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3),
                    nn.ReLU()
                ]
        ),
                channels=channels,
            )
        return modules

    def input_conv(self):
        modules = []
        depth_data = VGG_ENCODER_DATA[0]
        modules.append(
            nn.Conv2d(depth_data["first_conv"][0], depth_data["first_conv"][1],
                      kernel_size=1))
        return modules

    def output_conv(self, depth):
        modules = []
        depth_data = VGG_ENCODER_DATA[depth]
        modules.extend(self.conv_block(depth_data["first_conv"]))
        return modules

    def depth_level(self, channels: Sequence[int]):
        modules = []
        modules.extend(self.conv_block(channels))
        modules.append(nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True))
        return modules

    def build_model(self, name: str, layer: int) -> None:
        modules = []
        modules.extend(self.input_conv())

        for depth in range(1, layer):
            depth_data = VGG_ENCODER_DATA[depth]
            modules.extend(self.conv_block(depth_data["first_conv"]))
            modules.extend(self.depth_level(depth_data["channels"]))

        modules.extend(self.output_conv(layer))

        self.models[name] = enc.SequentialEncoder(modules)

    def load_models(self, layers: Optional[Sequence[int]] = None, init_weights: bool = True) -> enc.MultiLayerEncoder:
        if layers is None:
            layers = [len(VGG_ENCODER_DATA.keys()) - 1]

        for layer in layers:
            vgg_data = VGG_ENCODER_DATA[layer]
            self.build_model(vgg_data["name"], layer)
            if init_weights:
                self.init_model(vgg_data["filename"], vgg_data["name"])

            return self._multi_layer_encoder(self.models[vgg_data["name"]])

    def _multi_layer_encoder(self, encoder: enc.SequentialEncoder) -> enc.MultiLayerEncoder:
        modules = []
        block = 1
        depth = 0
        for module in encoder._modules.values():
            if isinstance(module, nn.Conv2d):
                name = f"conv{block}_{depth}"
                if depth == 0:
                    depth += 1
            elif isinstance(module, nn.BatchNorm2d):
                name = f"bn{block}_{depth}"
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

class EncoderVGGModels(PretrainedVGGModels):
    def download_models(self):
        for id, filename in enumerate(ENCODER_FILES, 1):
            self.download(id, filename)

def vgg_multi_layer_encoder() -> enc.MultiLayerEncoder:
    here = path.dirname(__file__)
    model_dir = path.join(here, "models")
    loader = VGGEncoderLoader(model_dir)
    vgg_decoder = EncoderVGGModels(model_dir, loader=loader)
    return vgg_decoder.load_models()

