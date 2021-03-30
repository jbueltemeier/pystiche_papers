from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

import pystiche
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale

from ..utils import AutoPadConv2d, AutoPadConvTranspose2d, ResidualBlock

__all__ = [
    "conv",
    "conv_block",
    "residual_block",
    "encoder",
    "decoder",
    "Transformer",
]


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    cls: Union[Type[nn.Conv2d], Type[nn.ConvTranspose2d]]
    kwargs: Dict[str, Any]
    if padding is None:
        cls = AutoPadConvTranspose2d if upsample else AutoPadConv2d
        kwargs = {}
    else:
        cls = nn.ConvTranspose2d if upsample else nn.Conv2d
        kwargs = dict(padding=padding)
    return cls(in_channels, out_channels, kernel_size, stride=stride, **kwargs)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
    relu: bool = True,
    inplace: bool = True,
) -> nn.Sequential:
    modules: List[nn.Module] = [
        conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
        ),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
    ]
    if relu:
        modules.append(nn.ReLU(inplace=inplace))
    return nn.Sequential(*modules)


def residual_block(channels: int, inplace: bool = True) -> ResidualBlock:
    in_channels = out_channels = channels
    kernel_size = 3
    residual = nn.Sequential(
        conv_block(in_channels, out_channels, kernel_size, stride=1, inplace=inplace,),
        conv_block(in_channels, out_channels, kernel_size, stride=1, relu=False,),
    )

    return ResidualBlock(residual)


def encoder() -> pystiche.SequentialModule:
    modules = (
        conv_block(in_channels=1, out_channels=32, kernel_size=9,),
        conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=2,),
        conv_block(in_channels=64, out_channels=128, kernel_size=3, stride=2,),
        residual_block(channels=128),
        residual_block(channels=128),
        residual_block(channels=128),
        residual_block(channels=128),
        residual_block(channels=128),
    )
    return pystiche.SequentialModule(*modules)


def decoder() -> pystiche.SequentialModule:
    class ValueRangeDelimiter(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x)

    modules = (
        conv_block(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, upsample=True,
        ),
        conv_block(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, upsample=True,
        ),
        AutoPadConv2d(in_channels=32, out_channels=1, kernel_size=9,),
        ValueRangeDelimiter(),
    )

    return pystiche.SequentialModule(*modules)


class Transformer(nn.Module):
    def __init__(self, fakegrayscale: bool = True) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.fakegrayscale = fakegrayscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = cast(torch.Tensor, self.decoder(self.encoder(x)))
        if not self.fakegrayscale:
            return output
        return cast(torch.Tensor, grayscale_to_fakegrayscale(output))


def transformer() -> Transformer:
    return Transformer()
