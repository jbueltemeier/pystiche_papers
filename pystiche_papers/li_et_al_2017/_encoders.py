from os import path
from typing import List, Optional, Sequence, Tuple, cast, Dict, Union

from torch import nn

from pystiche import enc

from ._utils import ModelLoader, PretrainedVGGModels

__all__ = ["VGGEncoderLoader", "vgg_multi_layer_encoder"]

BASE_URL = (
    "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"
)

ENCODER_FILE = "vgg_normalised_conv5_1.pth"

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg19': ['R', 64, 'R', 64, 'M', 'R', 128, 'R', 128, 'M', 'R', 256, 'R', 256, 'R', 256, 'R', 256, 'M', 'R', 512, 'R', 512, 'R', 512, 'R', 512, 'M', 'R', 512],
}


class VGGEncoderBuilder(object):
    def __init__(self) -> None:
        super().__init__()

    def conv_block(self, in_channels: int, out_channels: int) -> Sequence[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU()
        ]

    def input_conv(self) -> nn.Module:
        return nn.Conv2d(3, 3, kernel_size=1)

    def build_model(self) -> enc.SequentialEncoder:
        modules: List[nn.Module] = []
        modules.append(self.input_conv())

        in_channels = 3
        for cfg in cfgs["vgg19"]:
            if isinstance(cfg, int):
                modules.extend(self.conv_block(in_channels, cfg))
                in_channels = cfg
            elif cfg == "R":
                modules.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            else:  # isinstance(cfg, str) ('M')
                modules.append(nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True))

        return enc.SequentialEncoder(modules)


class VGGEncoderLoader(ModelLoader):
    def __init__(self, root: str,) -> None:
        super().__init__(root=root)
        self.builder = VGGEncoderBuilder()

    def build_model(self, name: str) -> None:  # type: ignore[override]
        self.models[name] = self.builder.build_model()

    def load_models(self, init_weights: bool = True) -> enc.MultiLayerEncoder:
        self.build_model("vgg19")
        if init_weights:
            self.init_model(ENCODER_FILE, "vgg19")

        return self._multi_layer_encoder(self.models["vgg19"])

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
