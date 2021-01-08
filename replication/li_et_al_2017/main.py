import os
from argparse import Namespace
from os import path

import pystiche_papers.li_et_al_2017 as paper
from pystiche import image, misc, optim
from pystiche_papers.li_et_al_2017._modules import wct_transformer
from pystiche_papers.li_et_al_2017._utils import hyper_parameters as _hyper_parameters


def training(args):
    contents = ("women1", "flower", "vincent", "women2", "women3", "bridge", "tubingen")
    styles = (
        "abstract",
        "water",
        "tiger",
        "iron_art",
        "women_painting",
        "antimonocromatismo",
        "brick",
        "brick1",
        "seated_nude",
        "women_hat",
        "women_dress",
    )

    image_size = 512

    images = paper.images()
    images.download(args.image_source_dir)

    hyper_parameters = _hyper_parameters(impl_params=args.impl_params)
    for style in styles:
        style_image = images[style].read(
            device=args.device, size=(image_size, image_size)
        )

        transformer = wct_transformer(
            impl_params=args.impl_params, hyper_parameters=hyper_parameters
        )
        transformer = transformer.to(args.device)
        transformer.set_target_image(style_image)

        for content in contents:
            content_image = images[content].read(
                device=args.device, size=(image_size, image_size)
            )
            output_image = paper.stylization(
                content_image, transformer, impl_params=args.impl_params,
            )

            output_name = f"{style}_{content}"
            if args.impl_params:
                output_name += "__impl_params"

            output_name += str(image_size)
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    device = None
    impl_params = True
    quiet = False

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "data", "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "data", "images", "results")
    image_results_dir = process_dir(image_results_dir)

    device = misc.get_device(device=device)
    logger = optim.OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        impl_params=impl_params,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    training(args)
