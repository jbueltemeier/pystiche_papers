import bisect
from typing import List, Tuple

import torch

from pystiche.image.utils import extract_num_channels

__all__ = [
    "wct",
]


def center_feature_maps(enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    channels = extract_num_channels(enc)
    enc = enc.view(channels, -1)
    mean_enc = torch.mean(enc, 1).unsqueeze(1).expand_as(enc)
    return enc - mean_enc, mean_enc


def maybe_reduce_channels(
        channels: int,
        eigenvalues: List[int],
        reduce_channels: bool = True,
        eps: float = 0.00001,
) -> int:
    return (
        channels - bisect.bisect_left(eigenvalues, eps) if reduce_channels else channels
    )


def get_linear_transform(
        enc: torch.Tensor,
        reduce_channels: bool = True,
        inverse: bool = False,
        eps: float = 0.00001,
) -> torch.Tensor:
    cov = torch.mm(enc, enc.t()).div(enc.size()[1] - 1)
    _, s, v = torch.svd(cov, some=False)
    reduced_channels = maybe_reduce_channels(
        enc.size()[1], list(s), reduce_channels=reduce_channels, eps=eps
    )
    d = (
        (s[0:reduced_channels]).pow(0.5)
        if inverse
        else (s[0:reduced_channels]).pow(-0.5)
    )
    transform = torch.mm(v[:, 0:reduced_channels], torch.diag(d))
    transform = torch.mm(transform, (v[:, 0:reduced_channels].t()))
    return transform


def wct(
        content_enc: torch.Tensor,
        style_enc: torch.Tensor,
        alpha: float,
        reduce_channels: bool = True,
) -> torch.Tensor:
    enc, _ = center_feature_maps(content_enc)
    linear_transform = get_linear_transform(enc, reduce_channels=reduce_channels)
    whitened_enc = torch.mm(linear_transform, enc)
    style_enc, mean_style_enc = center_feature_maps(style_enc)
    style_linear_transform = get_linear_transform(
        style_enc, reduce_channels=reduce_channels, inverse=True
    )
    mean_colored_enc = torch.mm(style_linear_transform, whitened_enc)
    colored_enc = mean_colored_enc + mean_style_enc
    colored_enc = colored_enc.view_as(content_enc)
    return alpha * colored_enc + (1.0 - alpha) * content_enc
