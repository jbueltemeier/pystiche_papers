from typing import Optional, cast, Union

import torch
from torch import nn
from pystiche_papers.utils import HyperParameters

from ._modules import wct_transformer
from ._utils import hyper_parameters as _hyper_parameters

__all__ = [
    "stylization",
]


def stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, torch.Tensor],
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        transformer: Transformer for style transfer or the style_image that is used as
            target_image of the :func:`~pystiche_papers.li_et_al_2017.wct_transformer`.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_et_al_2017-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_et_al_2017.hyper_parameters` is used.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    device = input_image.device

    if isinstance(transformer, torch.Tensor):
        style_image = transformer
        transformer = wct_transformer(
            impl_params=impl_params, hyper_parameters=hyper_parameters
        )
        transformer = transformer.to(device)
        transformer.set_target_image(style_image)

    with torch.no_grad():
        output_image = transformer(input_image)

    return cast(torch.Tensor, output_image)
