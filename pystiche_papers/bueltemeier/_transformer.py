from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn

from pystiche import misc
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale
from pystiche_papers.ulyanov_et_al_2016._modules import (
    HourGlassBlock,
    join_channelwise,
    noise,
)
from ._utils import hyper_parameters as _hyper_parameters


from ..utils import AutoPadConv2d, SequentialWithOutChannels

__all__ = [
    "ConvBlock",
    "ConvSequence",
    "JoinBlock",
    "BranchBlock",
    "level",
    "Transformer",
    "transformer",
]


class ConvBlock(SequentialWithOutChannels):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        inplace: bool = True,
    ) -> None:
        modules = (
            (
                "conv",
                AutoPadConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding_mode="reflect",
                ),
            ),
            (
                "norm",
                nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            ),
            ("act", nn.ReLU(inplace=inplace)),
        )
        super().__init__(OrderedDict(modules), out_channel_name="conv")


class ConvSequence(SequentialWithOutChannels):
    def __init__(
        self, in_channels: int, out_channels: int, inplace: bool = True,
    ):
        def conv_block(
            in_channels: int, out_channels: int, kernel_size: int
        ) -> ConvBlock:
            return ConvBlock(in_channels, out_channels, kernel_size, inplace=inplace,)

        modules = (
            ("conv_block1", conv_block(in_channels, out_channels, kernel_size=3)),
            ("conv_block2", conv_block(out_channels, out_channels, kernel_size=3)),
            ("conv_block3", conv_block(out_channels, out_channels, kernel_size=1)),
        )

        super().__init__(OrderedDict(modules))


class JoinBlock(nn.Module):
    def __init__(
        self,
        branch_in_channels: Sequence[int],
        names: Optional[Sequence[str]] = None,
        channel_dim: int = 1,
    ) -> None:
        super().__init__()

        num_branches = len(branch_in_channels)
        if names is None:
            names = [str(idx) for idx in range(num_branches)]
        else:
            if len(names) != num_branches:
                raise RuntimeError

        norm_modules = [
            nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True)
            for in_channels in branch_in_channels
        ]

        for name, module in zip(names, norm_modules):
            self.add_module(name, module)

        self.norm_modules = norm_modules
        self.channel_dim = channel_dim

    @property
    def out_channels(self) -> int:
        return sum([norm.num_features for norm in self.norm_modules])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return join_channelwise(
            *[norm(input) for norm, input in misc.zip_equal(self.norm_modules, inputs)],
            channel_dim=self.channel_dim,
        )


class BranchBlock(nn.Module):
    def __init__(
        self, deep_branch: nn.Module, shallow_branch: nn.Module,
    ):
        super().__init__()
        self.deep = deep_branch
        self.shallow = shallow_branch
        self.join = JoinBlock(
            (
                cast(int, deep_branch.out_channels),
                cast(int, shallow_branch.out_channels),
            ),
            ("deep", "shallow"),
        )

    @property
    def out_channels(self) -> int:
        return self.join.out_channels

    def forward(self, input: Any, **kwargs: Any) -> torch.Tensor:
        deep_output = self.deep(input, **kwargs)
        shallow_output = self.shallow(input, **kwargs)
        return cast(torch.Tensor, self.join(deep_output, shallow_output))


def level(
    prev_level_block: Optional[SequentialWithOutChannels],
    in_channels: int = 1,
    num_noise_channels: int = 0,
    inplace: bool = True,
) -> SequentialWithOutChannels:
    def conv_sequence(
        in_channels: int, out_channels: int, use_noise: bool = True
    ) -> SequentialWithOutChannels:
        modules: List[Tuple[str, nn.Module]] = []

        if use_noise:
            noise_module = noise(
                in_channels=in_channels, num_noise_channels=num_noise_channels,
            )
            in_channels = noise_module.out_channels
            modules.append(("noise", noise_module))

        conv_seq = ConvSequence(in_channels, out_channels, inplace=inplace,)

        if not use_noise:
            return conv_seq

        modules.append(("conv_seq", conv_seq))
        return SequentialWithOutChannels(OrderedDict(modules))

    shallow_branch = conv_sequence(in_channels, out_channels=8)

    if prev_level_block is None:
        return shallow_branch

    deep_branch = HourGlassBlock(prev_level_block)
    branch_block = BranchBlock(deep_branch, shallow_branch)

    output_conv_seq = conv_sequence(
        branch_block.out_channels, branch_block.out_channels
    )

    return SequentialWithOutChannels(
        OrderedDict((("branch", branch_block), ("output_conv_seq", output_conv_seq)))
    )


class Transformer(nn.Sequential):
    def __init__(self, levels: int,) -> None:
        class ValueRangeDelimiter(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.tanh(x)

        class FakegrayscaleOutput(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return cast(torch.Tensor, grayscale_to_fakegrayscale(x))

        pyramid = None
        for _ in range(levels):
            pyramid = level(pyramid)

        output_conv = cast(
            Union[nn.Conv2d, ConvBlock],
            ConvBlock(
                cast(int, cast(SequentialWithOutChannels, pyramid).out_channels),
                out_channels=1,
                kernel_size=1,
                stride=1,
            ),
        )

        super().__init__(
            OrderedDict(
                cast(
                    Tuple[Tuple[str, nn.Module]],
                    (
                        ("image_pyramid", pyramid),
                        ("output_conv", output_conv),
                        ("ValueRangeDelimiter", ValueRangeDelimiter()),
                        ("FakegrayscaleOutput", FakegrayscaleOutput()),
                    ),
                )
            )
        )


def transformer(style: Optional[str] = None, levels: Optional[int] = None) -> Transformer:
    r"""Transformer from :cite:`ULVL2016,UVL2017`.

    Args:
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`~pystiche_papers.ulyanov_et_al_2016.images`. If omitted, the
            transformer is initialized with random weights.
        levels: Number of levels in the transformer. Defaults to ``6``.

    """

    if levels is None:
        hyper_parameters = _hyper_parameters()
        levels = hyper_parameters.transformer.levels
    return Transformer(levels)
