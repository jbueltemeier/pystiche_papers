from abc import abstractmethod
from typing import Callable, Dict, Optional, Sequence, Union, cast

import torch

import pystiche
from pystiche import enc
from pystiche_papers.utils import HyperParameters

from ._decoders import SequentialDecoder, vgg_decoders
from ._encoders import vgg_multi_layer_encoder
from ._transform import wct
from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["WCTAutoEncoder", "TransformAutoEncoderContainer"]


class _AutoEncoder(pystiche.Module):
    r"""Abstract base class for all Autoencoders.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        decoder: Decoder that is used to decode the encodings to an output image.
    """

    def __init__(self, encoder: enc.Encoder, decoder: SequentialDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def input_image_to_enc(self, image: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(image))

    def enc_to_output_image(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(enc))

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image)


class _TransformAutoEncoder(_AutoEncoder):
    r"""Abstract base class for all Autoencoders transforming in an encoded space."""
    target_enc: torch.Tensor

    def __init__(self, encoder: enc.Encoder, decoder: SequentialDecoder) -> None:
        super().__init__(encoder, decoder)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        target_enc = self.target_enc
        input_enc = self.input_image_to_enc(image)
        transformed_enc = self.transform(input_enc, target_enc)
        return self.enc_to_output_image(transformed_enc)

    @abstractmethod
    def transform(self, enc: torch.Tensor, target_enc: torch.Tensor) -> torch.Tensor:
        pass

    def set_target_image(self, image: torch.Tensor) -> None:
        self.register_buffer("target_image", image)
        with torch.no_grad():
            enc = self.input_image_to_enc(image)
        self.register_buffer("target_enc", enc)

    @property
    def has_target_image(self) -> bool:
        return "target_image" in self._buffers


class WCTAutoEncoder(_TransformAutoEncoder):
    r"""Autoencoder that uses the WCT from :cite:`Li2017` on the encodings.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        decoder: Decoder that is used to decode the encodings to an output image.
        weight: Weight to control the strength of stylization.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_et_al_2017-impl_params>`.
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        decoder: SequentialDecoder,
        weight: float = 1.0,
        impl_params: bool = True,
    ) -> None:
        self.reduce_channels = impl_params
        self.weight = weight
        super().__init__(encoder, decoder)

    def transform(self, enc: torch.Tensor, target_enc: torch.Tensor) -> torch.Tensor:
        return wct(enc, target_enc, self.weight, reduce_channels=self.reduce_channels)


class TransformAutoEncoderContainer(pystiche.Module):
    r"""Generic container for :class:`_TransformAutoEncoder` s.

    If called with an image passes it sequentially to all operators and returns an
    output_image.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        decoders: Decoders that are used to decode the encodings to an output image at
            different levels.
        get_autoencoder: Callable that returns an TransformAutoEncoder given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder``, a corresponding decoder and its corresponding
            weight.
        level_weights: Weights of the Autoencoders passed to ``get_autoencoder``. If
            sequence of ``float``s its length has to match ``layers``. Defaults to
            ``1.0`.
    """

    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        decoders: Dict[
            str, SequentialDecoder
        ],  # TODO: Order is important. Add sort and parameter reverse order?
        get_autoencoder: Callable[
            [enc.Encoder, SequentialDecoder, float], _TransformAutoEncoder
        ],
        level_weights: Union[float, Sequence[float]] = 1.0,
    ) -> None:
        if type(level_weights) == float:
            level_weights = cast(Sequence[float], [level_weights] * len(decoders))

        def get_single_autoencoder(layer: str, weight: float) -> _TransformAutoEncoder:
            encoder = multi_layer_encoder.extract_encoder(layer)
            decoder = decoders[layer]
            return get_autoencoder(encoder, decoder, weight)

        named_autoencoder = [
            (layer, get_single_autoencoder(layer, weight),)
            for layer, weight in zip(decoders.keys(), cast(Sequence, level_weights))
        ]
        super().__init__()
        self.add_named_modules(named_autoencoder)

    def process_input_image(self, input_image: torch.Tensor) -> torch.Tensor:
        output_image = input_image
        for _, autoencoder in self.named_children():
            output_image = autoencoder(output_image)
        return output_image

    def set_target_image(self, image: torch.Tensor) -> None:
        for _, autoencoder in self.named_children():
            cast(_TransformAutoEncoder, autoencoder).set_target_image(image)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image)


def wct_transformer(
    impl_params: bool = True, hyper_parameters: Optional[HyperParameters] = None
) -> TransformAutoEncoderContainer:
    r"""Multi Layer whitening and coloring transformer from :cite:`Li2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_et_al_2017-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_et_al_2017.hyper_parameters` is used.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    decoders = vgg_decoders(hyper_parameters=hyper_parameters)
    multi_layer_encoder = vgg_multi_layer_encoder()

    level_weights = hyper_parameters.transform.weight

    def get_autoencoder(
        encoder: enc.Encoder, decoder: SequentialDecoder, weight: float
    ) -> WCTAutoEncoder:
        return WCTAutoEncoder(encoder, decoder, weight=weight, impl_params=impl_params)

    return TransformAutoEncoderContainer(
        multi_layer_encoder, decoders, get_autoencoder, level_weights=level_weights
    )
