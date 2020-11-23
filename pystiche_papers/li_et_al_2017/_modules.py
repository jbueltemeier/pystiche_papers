import bisect

import torch

from pystiche.image.utils import extract_image_size, extract_num_channels


def whitening(
    features: torch.Tensor, impl_params: bool = True, eps: float = 0.00001
) -> torch.Tensor:
    width, height = extract_image_size(features)
    mean_features = features - torch.mean(features, 2)

    cov = torch.mm(mean_features, mean_features.t()).div((width * height) - 1)
    u, s, v = torch.svd(cov, some=False)

    reduced_channels = (
        bisect.bisect_left(list(s), eps)
        if impl_params
        else extract_num_channels(features)
    )
    d = (s[0:reduced_channels]).pow(-0.5)

    transform = torch.mm(v[:, 0:reduced_channels], torch.diag(d))
    transform = torch.mm(transform, (v[:, 0:reduced_channels].t()))
    return torch.mm(transform, mean_features)


def coloring(
    whitened_features: torch.Tensor,
    style_features: torch.Tensor,
    impl_params: bool = True,
    eps: float = 0.00001,
) -> torch.Tensor:
    style_width, style_height = extract_image_size(style_features)
    mean_style = torch.mean(style_features, 1)
    mean_features = style_features - mean_style

    style_cov = torch.mm(mean_features, mean_features.t()).div(
        (style_width * style_height) - 1
    )
    style_u, style_s, style_v = torch.svd(style_cov, some=False)

    reduced_channels = (
        bisect.bisect_left(list(style_s), eps)
        if impl_params
        else extract_num_channels(whitened_features)
    )
    style_d = (style_s[0:reduced_channels]).pow(0.5)

    transform = torch.mm(style_v[:, 0:reduced_channels], torch.diag(style_d))
    transform = torch.mm(transform, style_v[:, 0:reduced_channels].t())
    colored = torch.mm(transform, whitened_features)
    return colored + mean_style


def wct(
    content_features: torch.Tensor,
    style_features: torch.Tensor,
    alpha: float,
    impl_params: bool = True,
) -> torch.Tensor:
    whitened_features = whitening(content_features, impl_params=impl_params)
    colored_features = coloring(
        whitened_features, style_features, impl_params=impl_params
    )
    return alpha * colored_features + (1.0 - alpha) * content_features
