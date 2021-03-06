import pytest

import pystiche_papers.gatys_et_al_2017 as paper


@pytest.fixture(scope="package")
def vgg_load_weights_mock(package_mocker):
    return package_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder_mock(package_mocker, vgg_load_weights_mock):
    multi_layer_encoder = paper.multi_layer_encoder()

    def new():
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return package_mocker.patch(
        "pystiche_papers.gatys_et_al_2017._loss._multi_layer_encoder", new,
    )
