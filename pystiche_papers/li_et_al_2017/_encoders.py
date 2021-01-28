from os import path
from typing import Dict, List, Sequence, Union, cast

from torch import nn

from pystiche import enc

from ._utils import ModelLoader, PretrainedVGGModels

__all__ = ["VGGEncoderLoader", "vgg_multi_layer_encoder"]

BASE_URL = (
    "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"
)

ENCODER_FILE: Dict[str, str] = {"relu5_1": "vgg_normalised_conv5_1.pth"}

cfgs: List[Union[str, int]] = [
    "R",
    64,
    "R",
    64,
    "M",
    "R",
    128,
    "R",
    128,
    "M",
    "R",
    256,
    "R",
    256,
    "R",
    256,
    "R",
    256,
    "M",
    "R",
    512,
    "R",
    512,
    "R",
    512,
    "R",
    512,
    "M",
    "R",
    512,
]


class VGGEncoderBuilder(object):
    def __init__(self) -> None:
        super().__init__()

    def conv_block(self, in_channels: int, out_channels: int) -> Sequence[nn.Module]:
        return [nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU()]

    def input_conv(self) -> nn.Module:
        return nn.Conv2d(3, 3, kernel_size=1)

    def build_model(self) -> enc.SequentialEncoder:
        # TODO: use the collect_modules method with state_dict_maps when the new
        #  pystiche version ist integrated
        modules: List[nn.Module] = []
        modules.append(self.input_conv())

        in_channels = 3
        for cfg in cfgs:
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

    def load_model(
        self, layer: Union[int, str], init_weights: bool = True
    ) -> enc.MultiLayerEncoder:
        if layer != "relu5_1":
            msg = (
                f"You are using layer {layer}, this is not integrated. Please "
                f"use 'relu5_1' as this will be loaded as MultiLayerEncoder."
            )
            raise ValueError(msg)

        model = self.builder.build_model()
        if init_weights:
            model = self.init_model(model, ENCODER_FILE[cast(str, layer)])  # type: ignore[assignment]

        return self._multi_layer_encoder(model)

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
        id = 5
        layer = "relu5_1"
        self.download(id, ENCODER_FILE[layer])

    def load_models(self) -> enc.MultiLayerEncoder:
        return cast(VGGEncoderLoader, self.loader).load_model("relu5_1")


def vgg_multi_layer_encoder(
    framework: str = "UniversalStyleTransfer",
) -> enc.MultiLayerEncoder:
    if framework == "UniversalStyleTransfer":
        here = path.dirname(__file__)
        model_dir = path.join(here, "models")
        loader = VGGEncoderLoader(model_dir)
        encoder_model = EncoderVGGModel(model_dir, loader=loader)
        return encoder_model.load_models()
    elif framework == "caffe":
        return enc.vgg19_multi_layer_encoder(weights="caffe")
    else:
        msg = (
            f"The framework {framework} is not integrated. Please use 'caffe' or "
            f"'UniversalStyleTransfer'."
        )
        raise ValueError(msg)
