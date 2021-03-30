from typing import Optional

from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "gram_style_loss",
    "mrf_style_loss",
    "perceptual_loss",
]


def content_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.FeatureReconstructionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return ops.FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


def gram_style_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight)

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.gram_style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.gram_style_loss.layer_weights,
        score_weight=hyper_parameters.gram_style_loss.score_weight,
    )


def mrf_style_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.MRFOperator:
        return ops.MRFOperator(
            encoder,
            hyper_parameters.mrf_style_loss.patch_size,  # type: ignore[union-attr]
            stride=hyper_parameters.mrf_style_loss.stride,  # type: ignore[union-attr]
            score_weight=layer_weight,
        )

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.mrf_style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.mrf_style_loss.layer_weights,
        score_weight=hyper_parameters.mrf_style_loss.score_weight,
    )


def perceptual_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    # style_loss = ops.OperatorContainer(
    #     [
    #         (
    #             "mrf_loss",
    #             mrf_style_loss(
    #                 multi_layer_encoder=multi_layer_encoder,
    #                 hyper_parameters=hyper_parameters,
    #             ),
    #         ),
    #         (
    #             "gram_loss",
    #             gram_style_loss(
    #                 multi_layer_encoder=multi_layer_encoder,
    #                 hyper_parameters=hyper_parameters,
    #             ),
    #         ),
    #     ]
    # )

    return loss.PerceptualLoss(
        content_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
        gram_style_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters
        ),
    )
