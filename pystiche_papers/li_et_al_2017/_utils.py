import os
from abc import abstractmethod
from os import path
from typing import Callable, Dict, List, Optional, Sequence, TypeVar, Union
from urllib.parse import urljoin

import more_itertools

import torch
from torch.hub import load_state_dict_from_url

from pystiche import enc
from pystiche_papers.utils import HyperParameters

__all__ = [
    "channel_progression",
    "ModelLoader",
]

BASE_URL = (
    "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"
)

T = TypeVar("T")


def channel_progression(
    module_fn: Callable[[int, int], T], channels: Sequence[int]
) -> List[T]:
    return [
        module_fn(*channels_pair) for channels_pair in more_itertools.pairwise(channels)
    ]


class ModelLoader(object):
    def __init__(self, root: str) -> None:
        self.root = root
        self.models: Dict[str, enc.SequentialEncoder] = {}

    def model_file_path(self, filename: str) -> str:
        return os.path.join(self.root, filename)

    def init_model(self, filename: str, name: str) -> None:
        self.models[name].load_state_dict(torch.load(self.model_file_path(filename)))
        self.models[name].eval()

    @abstractmethod
    def build_model(self, depth: int) -> None:
        pass

    @abstractmethod
    def load_models(
        self, init_weights: bool = True
    ) -> Union[Dict[str, enc.SequentialEncoder], enc.MultiLayerEncoder]:
        pass


class PretrainedVGGModels(object):
    def __init__(
        self,
        root: str,
        loader: ModelLoader,
        layers: Optional[Sequence[int]] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.layers = layers

        self.loader = loader

        if download:
            self.download_models()

        self.load_models()
        super().__init__()

    def url(self, id: int, filename: str) -> str:
        path = f"vgg19_{id:01d}//{filename}"
        return urljoin(BASE_URL, path)

    def download(self, id: int, filename: str) -> None:
        root_url = self.url(id, filename)
        if path.exists(root_url):
            msg = (
                f"The model directory {root_url} already exists. If you want to "
                "re-download the model, delete the file."
            )
            raise RuntimeError(msg)

        load_state_dict_from_url(root_url, model_dir=self.root)

    @abstractmethod
    def download_models(self) -> None:
        pass

    @abstractmethod
    def load_models(
        self,
    ) -> Union[Dict[str, enc.SequentialEncoder], enc.MultiLayerEncoder]:
        pass


def hyper_parameters(impl_params: bool = True) -> HyperParameters:
    r"""Hyper parameters from :cite:`Li2017`."""
    return HyperParameters(
        transform=HyperParameters(weight=0.6 if impl_params else 1.0,),
        decoder=HyperParameters(layers=[5, 4, 3, 2, 1],),
    )
