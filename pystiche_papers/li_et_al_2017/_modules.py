from abc import abstractmethod
from typing import Callable, Sequence, Tuple, Union, cast

import torch

import pystiche
import pystiche_papers.li_et_al_2017 as paper
from pystiche import enc

__all__ = ["WCTAutoEncoder", "TransformAutoEncoderContainer"]


class _AutoEncoder(pystiche.Module):
    def __init__(self, encoder: enc.Encoder, decoder: enc.Encoder,) -> None:
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

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image)


class _TransformAutoEncoder(_AutoEncoder):
    target_enc: torch.Tensor

    def __init__(
        self, encoder: enc.Encoder, decoder: enc.Encoder, weight: float = 0.6,
    ) -> None:
        super().__init__(encoder, decoder)
        self.weight = weight

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
    def __init__(
        self,
        encoder: enc.Encoder,
        decoder: enc.Encoder,
        weight: float = 0.6,
        impl_params: bool = True,
    ) -> None:
        self.reduce_channels = impl_params
        super().__init__(encoder, decoder, weight=weight)

    def transform(self, enc: torch.Tensor, target_enc: torch.Tensor) -> torch.Tensor:
        return paper.wct(
            enc, target_enc, self.weight, reduce_channels=self.reduce_channels
        )


class TransformAutoEncoderContainer(pystiche.Module):
    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        decoders: Sequence[
            Tuple[str, enc.Encoder]
        ],  # TODO: Order is important. Add sort and parameter reverse order?
        get_autoencoder: Callable[
            [enc.Encoder, enc.Encoder, float], _TransformAutoEncoder
        ],
        level_weights: Union[float, Sequence[float]] = 0.6,
    ) -> None:
        if type(level_weights) == float:
            level_weights = cast(Sequence[float], [level_weights] * len(decoders))

        def get_single_autoencoder(
            layer: str, decoder: enc.Encoder, weight: float
        ) -> _TransformAutoEncoder:
            encoder = multi_layer_encoder.extract_encoder(layer)
            return get_autoencoder(encoder, decoder, weight)

        named_autoencoder = [
            (name, get_single_autoencoder(name, decoder, weight))
            for (name, decoder), weight in zip(decoders, cast(Sequence, level_weights))
        ]
        super().__init__()
        self.add_named_modules(named_autoencoder)

    def process_input_image(self, input_image: torch.Tensor) -> torch.Tensor:
        output_image = input_image
        for _, autoencoder in self.named_children():
            autoencoder(output_image)
        return output_image

    def set_target_image(self, image: torch.Tensor) -> None:
        for autoencoder in self.children():
            cast(_TransformAutoEncoder, autoencoder).set_target_image(image)

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image)


def wct_transformer(impl_params: bool = True) -> TransformAutoEncoderContainer:
    multi_layer_encoder = enc.vgg19_multi_layer_encoder()
    # TODO: set the right decoders
    decoder1 = enc.SequentialEncoder([torch.nn.Conv2d(3, 3, 1),])
    decoder2 = enc.SequentialEncoder(
        [torch.nn.Conv2d(3, 3, 1), torch.nn.Conv2d(3, 3, 1),]
    )

    decoders = [("conv2_1", decoder2), ("conv1_1", decoder1)]

    def get_autoencoder(
        encoder: enc.Encoder, decoder: enc.Encoder, weight: float
    ) -> WCTAutoEncoder:
        return WCTAutoEncoder(encoder, decoder, weight=weight, impl_params=impl_params)

    return TransformAutoEncoderContainer(multi_layer_encoder, decoders, get_autoencoder)
