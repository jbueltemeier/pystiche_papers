from typing import cast

import torch

from ._modules import wct_transformer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = [
    "stylization",
]


def stylization(
    input_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        style_image: Image from which the style is taken.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see ??.

    """
    device = input_image.device

    preprocessor = _preprocessor()
    preprocessor = preprocessor.to(device)
    postprocessor = _postprocessor()
    postprocessor = postprocessor.to(device)

    transformer = wct_transformer(impl_params=impl_params)
    transformer = transformer.to(device)
    transformer.eval()

    transformer.set_target_image(preprocessor(style_image))

    with torch.no_grad():
        output_image = transformer(preprocessor(input_image))

    return cast(torch.Tensor, postprocessor(output_image.detach()))
