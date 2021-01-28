import bisect

import torch

from pystiche.image.utils import extract_image_size, extract_num_channels

__all__ = [
    "wct",
]


def whitening(
    enc: torch.Tensor, reduce_channels: bool = True, eps: float = 0.00001
) -> torch.Tensor:
    channels = extract_num_channels(enc)
    width, height = extract_image_size(enc)
    enc = enc.view(channels, -1)
    mean_enc = enc - torch.mean(enc, 1).unsqueeze(1).expand_as(enc)

    cov = torch.mm(mean_enc, mean_enc.t()).div((width * height) - 1)
    _, s, v = torch.svd(cov, some=False)

    reduced_channels = (
        channels - bisect.bisect_left(list(s), eps) if reduce_channels else channels
    )
    d = (s[0:reduced_channels]).pow(-0.5)

    transform = torch.mm(v[:, 0:reduced_channels], torch.diag(d))
    transform = torch.mm(transform, (v[:, 0:reduced_channels].t()))
    return torch.mm(transform, mean_enc)


def coloring(
    whitened_enc: torch.Tensor,
    style_enc: torch.Tensor,
    reduce_channels: bool = True,
    eps: float = 0.00001,
) -> torch.Tensor:
    if len(whitened_enc.shape) != 2:
        channels = extract_num_channels(whitened_enc)
        whitened_enc = whitened_enc.view(channels, -1)
    channels = extract_num_channels(style_enc)
    style_width, style_height = extract_image_size(style_enc)
    style_enc = style_enc.view(channels, -1)
    mean_style = torch.mean(style_enc, 1).unsqueeze(1).expand_as(style_enc)
    mean_enc = style_enc - mean_style

    style_cov = torch.mm(mean_enc, mean_enc.t()).div((style_width * style_height) - 1)
    _, style_s, style_v = torch.svd(style_cov, some=False)

    reduced_channels = (
        channels - bisect.bisect_left(list(style_s), eps)
        if reduce_channels
        else channels
    )
    style_d = (style_s[0:reduced_channels]).pow(0.5)

    transform = torch.mm(style_v[:, 0:reduced_channels], torch.diag(style_d))
    transform = torch.mm(transform, style_v[:, 0:reduced_channels].t())
    colored = torch.mm(transform, whitened_enc)
    return colored + mean_style


def wct(
    content_enc: torch.Tensor,
    style_enc: torch.Tensor,
    alpha: float,
    reduce_channels: bool = True,
) -> torch.Tensor:
    whitened_enc = whitening(content_enc, reduce_channels=reduce_channels)
    colored_enc = coloring(whitened_enc, style_enc, reduce_channels=reduce_channels)
    colored_enc = colored_enc.view_as(content_enc)
    return alpha * colored_enc + (1.0 - alpha) * content_enc
