import os
from abc import abstractmethod
from os import path
from typing import Dict, Optional, Sequence, Union
from urllib.parse import urljoin

import torch
from torch.hub import load_state_dict_from_url

import pystiche
from pystiche_papers.utils import HyperParameters

__all__ = [
    "ModelLoader",
]

BASE_URL = (
    "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"
)


class ModelLoader(object):
    def __init__(self, root: str) -> None:
        self.root = root

    def model_file_path(self, filename: str) -> str:
        return os.path.join(self.root, filename)

    def init_model(
        self, model: Union[pystiche.Module], filename: str
    ) -> pystiche.Module:
        model.load_state_dict(torch.load(self.model_file_path(filename)))
        model.eval()
        return model

    @abstractmethod
    def load_model(
        self, layer: Union[int, str], init_weights: bool = True
    ) -> pystiche.Module:
        pass


class PretrainedVGGModels(object):
    def __init__(
        self,
        root: str,
        loader: ModelLoader,
        layers: Optional[Sequence[Union[int, str]]] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))
        self.layers = layers

        self.loader = loader

        if download:
            self.download_models()

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
    ) -> Union[Dict[Union[int, str], pystiche.Module], pystiche.Module]:
        pass


def hyper_parameters(impl_params: bool = True) -> HyperParameters:
    r"""Hyper parameters from :cite:`Li2017`."""
    return HyperParameters(
        transform=HyperParameters(weight=0.6 if impl_params else 1.0,),
        decoder=HyperParameters(
            layers=["relu5_1", "relu4_1", "relu3_1", "relu2_1", "relu1_1"],
        ),
    )
