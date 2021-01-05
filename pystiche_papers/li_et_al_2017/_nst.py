from typing import cast, Optional

import torch

from pystiche_papers.utils import HyperParameters
from ._modules import wct_transformer
from ._utils import hyper_parameters as _hyper_parameters

__all__ = [
    "stylization",
]


def stylization(
    input_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        style_image: Image from which the style is taken.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_et_al_2017-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_et_al_2017.hyper_parameters` is used.

    """
    device = input_image.device

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    transformer = wct_transformer(impl_params=impl_params, hyper_parameters=hyper_parameters)
    transformer = transformer.to(device)

    with torch.no_grad():
        transformer.set_target_image(style_image)
        output_image = transformer(input_image)

    return cast(torch.Tensor, output_image)
