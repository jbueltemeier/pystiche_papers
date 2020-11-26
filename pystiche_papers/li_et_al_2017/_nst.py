from typing import Optional, cast

import torch

from ._modules import TransformAutoEncoderContainer, wct_transformer


def stylization(
    input_image: torch.Tensor,
    style_image: torch.Tensor,
    transformer: Optional[TransformAutoEncoderContainer] = None,
    impl_params: bool = True,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        style_image: Image from which the style is taken.
        transformer: Optional transformer for universal style transfer. If omitted, the
            default :func:`~pystiche_papers.li_et_al_2017.wct_transformer` is used.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see ??.

    """
    device = input_image.device
    if transformer is None:
        transformer = wct_transformer(impl_params=impl_params)

    transformer.set_target_image(style_image)
    transformer = transformer.to(device)

    with torch.no_grad():
        output_image = transformer(input_image)

    return cast(torch.Tensor, output_image.detach())
