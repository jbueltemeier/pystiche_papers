from typing import Optional, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from pystiche import loss, optim
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale
from pystiche_papers.utils import HyperParameters

from ..utils import batch_up_image
from ._data import content_transform as _content_transform
from ._data import style_transform as _style_transform
from ._loss import perceptual_loss
from ._modules import transformer as _transformer
from ._utils import hyper_parameters as _hyper_parameters
from ._utils import optimizer as _optimizer

__all__ = ["training", "stylization"]


def training(
    content_image_loader: DataLoader,
    style_image: torch.Tensor,
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image: Style image on which the ``transformer`` should be trained. If
            ``str``, the image is read from
            :func:`~pystiche_papers.bueltemeier_2021.images`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.bueltemeier_2021.hyper_parameters` is used.

    """
    device = style_image.device

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transformer = _transformer()
    transformer = transformer.train()
    transformer = transformer.to(device)

    criterion = perceptual_loss(hyper_parameters=hyper_parameters)
    criterion = criterion.eval()
    criterion = criterion.to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    # preprocessor = _preprocessor()
    # preprocessor = preprocessor.to(device)
    # style_image = preprocessor(style_image)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        input_image = grayscale_to_fakegrayscale(input_image)
        cast(loss.PerceptualLoss, criterion).set_content_image(input_image)

    return optim.default_transformer_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
    )


def stylization(input_image: torch.Tensor, transformer: nn.Module,) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transfromer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer.
    """
    device = input_image.device

    transformer = transformer.eval()
    transformer = transformer.to(device)

    transform = _content_transform()
    transform = transform.to(device)
    # preprocessor = _preprocessor()
    # preprocessor = preprocessor.to(device)

    # postprocessor = _postprocessor()
    # postprocessor = postprocessor.to(device)

    with torch.no_grad():
        input_image = transform(input_image)
        # input_image = preprocessor(input_image)
        output_image = transformer(input_image)
        # output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image).detach()
