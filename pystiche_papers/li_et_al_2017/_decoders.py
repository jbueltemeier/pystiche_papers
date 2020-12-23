from pystiche import enc
from torch import nn
from typing import Sequence,  Tuple, Optional, Dict
from os import path

from ._utils import ModelLoader, channel_progression, PretrainedVGGModels

__all__ = ["SequentialDecoder", "VGGDecoderLoader", "vgg_decoders"]


VGG_DECODER_DATA = {
        1: {
            "name": "conv1_1",
            "first_conv": (64, 3),
            "channels": (),
            "filename": "feature_invertor_conv1_1.pth"
        },
        2: {
            "name": "conv2_1",
            "first_conv": (128, 64),
            "channels": (64, 64),
            "filename": "feature_invertor_conv2_1.pth"
        },
        3: {
            "name": "conv3_1",
            "first_conv": (256, 128),
            "channels": (128, 128),
            "filename": "feature_invertor_conv3_1.pth"
        },
        4: {
            "name": "conv4_1",
            "first_conv": (512, 256),
            "channels": (256, 256, 256, 256),
            "filename": "feature_invertor_conv4_1.pth"
        },
        5: {
            "name": "conv5_1",
            "first_conv": (512, 512),
            "channels": (512, 512, 512, 512),
            "filename": "feature_invertor_conv5_1.pth"
        },
    }


class SequentialDecoder(enc.SequentialEncoder):
    r"""Decoder that operates in sequential manner.

    Args:
        modules: Sequential modules.
        layer: Name of the layer of the pre-trained network whose encodings can be decoded.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(modules=modules)



class VGGDecoderLoader(ModelLoader):
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

    def output_conv(self):
        modules = []
        depth_data = VGG_DECODER_DATA[1]
        modules.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        modules.append(
            nn.Conv2d(depth_data["first_conv"][0], depth_data["first_conv"][1],
                      kernel_size=3))
        return modules

    def depth_level(self, channels: Sequence[int]):
        modules = [nn.UpsamplingNearest2d(scale_factor=2)]
        modules.extend(self.conv_block(channels))
        return modules

    def build_model(self, name: str, layer: int) -> None:
        modules = []

        if layer > 1:
            for depth in range(layer, 1, -1):
                depth_data = VGG_DECODER_DATA[depth]
                modules.extend(self.conv_block(depth_data["first_conv"]))
                modules.extend(self.depth_level(depth_data["channels"]))

        modules.extend(self.output_conv())
        self.models[name] = SequentialDecoder(modules)

    def load_models(self, layers: Optional[Sequence[int]], init_weights: bool = True) -> Dict[str, enc.Encoder]:
        if layers is None:
            layers = VGG_DECODER_DATA.keys()

        for layer in layers:
            vgg_data = VGG_DECODER_DATA[layer]
            self.build_model(vgg_data["name"], layer)
            if init_weights:
                self.init_model(vgg_data["filename"], vgg_data["name"])
        return self.models


def vgg_decoders() -> Dict[str, SequentialDecoder]:
    here = path.dirname(__file__)

    model_dir = path.join(here, "models")
    loader = VGGDecoderLoader(model_dir)
    vgg_decoder = PretrainedVGGModels(model_dir, layers=[5, 4, 3, 2, 1], loader=loader)
    return vgg_decoder.load_models()






