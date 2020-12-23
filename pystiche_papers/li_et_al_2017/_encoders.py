from pystiche import enc
from torch import nn
from typing import Sequence, Tuple, Optional, Dict
from os import path

from ._utils import ModelLoader, channel_progression, PretrainedVGGModels

__all__ = ["VGGEncoderLoader", "vgg_encoders"]

BASE_URL = "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"

VGG_ENCODER_DATA = {
        0: {
            "name": "input_norm",
            "first_conv": (3, 3),
            "channels": (),
            "filename": ""
        },
        1: {
            "name": "conv1_1",
            "first_conv": (3, 64),
            "channels": (64, 64),
            "filename": "vgg_normalised_conv1_1.pth"
        },
        2: {
            "name": "conv2_1",
            "first_conv": (64, 128),
            "channels": (128, 128),
            "filename": "vgg_normalised_conv2_1.pth"
        },
        3: {
            "name": "conv3_1",
            "first_conv": (128, 256),
            "channels": (256, 256, 256, 256),
            "filename": "vgg_normalised_conv3_1.pth"
        },
        4: {
            "name": "conv4_1",
            "first_conv": (256, 512),
            "channels": (512, 512, 512, 512),
            "filename": "vgg_normalised_conv4_1.pth"
        },
        5: {
            "name": "conv5_1",
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

    def load_models(self, layers: Optional[Sequence[int]], init_weights: bool = True) -> None:
        if layers is None:
            layers = VGG_ENCODER_DATA.keys()

        for layer in layers:
            vgg_data = VGG_ENCODER_DATA[layer]
            self.build_model(vgg_data["name"], layer)
            if init_weights:
                self.init_model(vgg_data["filename"], vgg_data["name"])
        return self.models

def vgg_encoders() -> Dict[str, enc.SequentialEncoder]:
    here = path.dirname(__file__)

    model_dir = path.join(here, "models")
    loader = VGGEncoderLoader(model_dir)
    vgg_decoder = PretrainedVGGModels(model_dir, layers=[5, 4, 3, 2, 1], loader=loader)
    return vgg_decoder.load_models()

