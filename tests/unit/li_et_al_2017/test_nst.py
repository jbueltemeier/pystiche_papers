import unittest.mock

import pytest

import pytorch_testing_utils as ptu

import pystiche_papers.li_et_al_2017 as paper


def make_patch_target(name):
    return ".".join(("pystiche_papers", "li_et_al_2017", "_nst", name))


def attach_method_mock(mock, method, **attrs):
    if "name" not in attrs:
        attrs["name"] = f"{mock.name}.{method}()"

    method_mock = unittest.mock.Mock(**attrs)
    mock.attach_mock(method_mock, method)


@pytest.fixture
def make_nn_module_mock(mocker):
    def make_nn_module_mock_(name=None, identity=False, **kwargs):
        attrs = {}
        if name is not None:
            attrs["name"] = name
        if identity:
            attrs["side_effect"] = lambda x: x
        attrs.update(kwargs)

        mock = mocker.Mock(**attrs)

        for method in ("eval", "to", "train"):
            attach_method_mock(mock, method, return_value=mock)

        return mock

    return make_nn_module_mock_


@pytest.fixture
def patcher(mocker):
    def patcher_(name, **kwargs):
        return mocker.patch(make_patch_target(name), **kwargs)

    return patcher_


@pytest.fixture
def transformer_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("wct_transformer", return_value=mock)
    return patch, mock


@pytest.fixture
def stylization(input_image, transformer_mocks, content_image):
    _, transformer = transformer_mocks

    def stylization_(input_image_=None, transformer_=content_image, **kwargs):
        if input_image_ is None:
            input_image_ = input_image

        output = paper.stylization(input_image_, transformer_, **kwargs)

        if isinstance(transformer_, str):
            transformer.assert_called_once()
            args, kwargs = transformer.call_args
        else:
            try:
                transformer_.assert_called_once()
                args, kwargs = transformer.call_args
            except AttributeError:
                args = kwargs = None

        return args, kwargs, output

    return stylization_


def test_stylization_smoke(stylization, input_image):
    _, _, output_image = stylization(input_image)
    ptu.assert_allclose(output_image, input_image, rtol=1e-6)


def test_stylization_device(
    transformer_mocks, stylization, input_image,
):
    stylization(input_image)

    _, mock = transformer_mocks
    mock = mock.to
    mock.assert_called_once_with(input_image.device)
