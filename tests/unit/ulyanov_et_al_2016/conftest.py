import pytest

import pystiche_papers.ulyanov_et_al_2016 as paper


@pytest.fixture(scope="package")
def vgg_load_weights_mock(package_mocker):
    return package_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder_mock(package_mocker, vgg_load_weights_mock):
    multi_layer_encoder = paper.multi_layer_encoder()

    def trim_mock(*args, **kwargs):
        pass

    multi_layer_encoder.trim = trim_mock

    def new(impl_params=None):
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return package_mocker.patch(
        "pystiche_papers.ulyanov_et_al_2016._loss._multi_layer_encoder", new,
    )
