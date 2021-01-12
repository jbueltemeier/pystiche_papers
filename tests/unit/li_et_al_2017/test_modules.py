import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche_papers.li_et_al_2017 as paper
from pystiche import enc


def test_AutoEncoder_call(input_image):
    class TestAutoEncoder(paper._AutoEncoder):
        def process_input_image(self, image: torch.Tensor):
            return self.enc_to_output_image(self.input_image_to_enc(image))

    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    decoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    autoencoder = TestAutoEncoder(encoder, decoder)

    actual = autoencoder(input_image)
    desired = decoder(encoder(input_image))
    ptu.assert_allclose(actual, desired)
