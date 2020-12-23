from typing import Sequence, Callable, List, TypeVar, Dict, Optional
import more_itertools
from abc import abstractmethod
import torch
import os
from os import path
from pystiche import enc
from torch.hub import load_state_dict_from_url
from urllib.parse import urljoin

__all__ = [
    "channel_progression",
    "ModelLoader",
]

BASE_URL = "https://github.com/pietrocarbo/deep-transfer/raw/master/models/autoencoder_vgg19/"

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
        self.models = {}

    def model_file_path(self, filename: str) -> str:
        return os.path.join(self.root, filename)

    def init_model(self, filename: str, name: str) -> None:
        self.models[name].load_state_dict(
            torch.load(self.model_file_path(filename)))
        self.models[name].eval()

    @abstractmethod
    def build_model(self, depth: int) -> None:
        pass

    @abstractmethod
    def load_models(self, layers: Optional[Sequence[int]], init_weights: bool = True) -> Dict[str, enc.Encoder]:
        pass


class PretrainedVGGModels(object):
    def __init__(self, root: str, layers: Sequence[int], loader: ModelLoader,  download: bool = False) -> None:
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

    def download_models(self, files):
        for id, filename in enumerate(files, 1):
            self.download(id, filename)

    def load_models(self) -> Dict[str, enc.Encoder]:
        return self.loader.load_models(self.layers)