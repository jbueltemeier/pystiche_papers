from abc import abstractmethod
from typing import Union, cast

import torch

import pystiche
import pystiche_papers.li_et_al_2017 as paper
from pystiche import enc

__all__ = [
    "WCTAutoEncoder",
]


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
    def transform(self, enc: torch.Tensor, target_enc: torch.Tensor) -> torch.Tensor:
        return paper.wct(enc, target_enc, self.weight)
