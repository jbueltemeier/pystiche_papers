from pystiche import enc
from torch import nn
from abc import abstractmethod
import os
import torch
from typing import Sequence, Callable, List, TypeVar, Tuple, Optional, Dict
from os import path
from torch.hub import load_state_dict_from_url
from urllib.parse import urljoin
import more_itertools

__all__ = ["SequentialDecoder", "DecoderLoader", "VGGDecoderLoader", "VGGDecoders", "vgg_decoders"]

BASE_URL = "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"

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

T = TypeVar("T")


def channel_progression(
    module_fn: Callable[[int, int], T], channels: Sequence[int]
) -> List[T]:
    return [
        module_fn(*channels_pair) for channels_pair in more_itertools.pairwise(channels)
    ]



class SequentialDecoder(enc.SequentialEncoder):
    r"""Decoder that operates in sequential manner.

    Args:
        modules: Sequential modules.
        layer: Name of the layer of the pre-trained network whose encodings can be decoded.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__(modules=modules)


class DecoderLoader(object):
    def __init__(self, root: str) -> None:
        self.root = root
        self.decoders = {}

    @property
    def get_decoders(self) -> Dict[str, SequentialDecoder]:
        return self.decoders

    @abstractmethod
    def init_decoder(self, depth: int) -> None:
        pass

    @abstractmethod
    def build_decoder(self, depth: int) -> None:
        pass

    @abstractmethod
    def load_decoders(self, depth: int) -> None:
        pass


class VGGDecoderLoader(DecoderLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root=root)

    def model_file_path(self, filename: str) -> str:
        return os.path.join(self.root, filename)

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

    def build_decoder(self, name: str, layer: int) -> None:
        modules = []

        if layer > 1:
            for depth in range(layer, 1, -1):
                depth_data = VGG_DECODER_DATA[depth]
                modules.extend(self.conv_block(depth_data["first_conv"]))
                modules.extend(self.depth_level(depth_data["channels"]))

        modules.extend(self.output_conv())
        self.decoders[name] = SequentialDecoder(modules)

    def init_decoder(self, depth: int) -> None:
        depth_data = VGG_DECODER_DATA[depth]
        self.decoders[depth_data["name"]].load_state_dict(torch.load(self.model_file_path(depth_data["filename"])))
        self.decoders[depth_data["name"]].eval()

    def load_decoders(self, layers: Optional[Sequence[int]], init_weights: bool = True) -> None:
        if layers is None:
            layers = VGG_DECODER_DATA.keys()

        for layer in layers:
            self.build_decoder(VGG_DECODER_DATA[layer]["name"], layer)
            if init_weights:
                self.init_decoder(layer)



class VGGDecoders(object):
    def __init__(self, root: str, layers: Sequence[int], download: bool = False) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.layers = layers

        if download:
            self.download_models()

        self.load_models()
        super().__init__()


    def url(self, id: int, filename: str) -> str:
        path = f"vgg19_{id:01d}//{filename}"
        return urljoin(BASE_URL, path)

    def download(self, id: int, filename: str) -> None:
        root_url = self.url(id, filename)
        if path.exists(root_url):
            msg = (
                f"The model directory {root_url} already exists. If you want to "
                "re-download the model, delete the file."
            )
            raise RuntimeError(msg)

        load_state_dict_from_url(root_url, model_dir=self.root)

    def download_models(self):
        # TODO: rework this wrong items and test this
        for id, data in enumerate(self.decoder_files.items(), 1):
            filename = data[1]
            self.download(id, filename)


    def load_models(self) -> Dict[str, SequentialDecoder]:
        decoder_loader = VGGDecoderLoader(self.root)
        decoder_loader.load_decoders(layers=self.layers)
        return decoder_loader.decoders


def vgg_decoders() -> Dict[str, SequentialDecoder]:
    here = path.dirname(__file__)

    model_dir = path.join(here, "models")
    vgg_decoder = VGGDecoders(model_dir, layers=[1, 2, 3, 4, 5])
    return vgg_decoder.load_models()






