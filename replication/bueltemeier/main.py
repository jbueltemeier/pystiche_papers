import os
from argparse import Namespace
from os import path

import pystiche_papers.bueltemeier as paper
from pystiche import image, misc
from pystiche_papers import utils


def training(args):
    contents = ("karya", "004", "04", "bueltemeier")
    styles = ("DM100", "MAD20", "Specimen0", "UHD20")

    dataset = paper.dataset(path.join(args.dataset_dir, "content"),)
    image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda")
    )

    images = paper.images(args.image_source_dir)

    for style in styles:
        style_image = images[style].read(device=args.device, size=512)

        transformer = paper.training(image_loader, style_image)

        model_name = f"bueltemeier_2021__{style}"
        utils.save_state_dict(transformer, model_name, root=args.model_dir)

        for content in contents:
            content_image = images[content].read(device=args.device)
            output_image = paper.stylization(content_image, transformer)

            output_name = f"{style}_{content}"
            output_file = path.join(args.image_results_dir, f"{output_name}.png")
            image.write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    dataset_dir = None
    model_dir = None
    device = None

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

    if dataset_dir is None:
        dataset_dir = path.join(here, "data", "images", "dataset")
    dataset_dir = process_dir(dataset_dir)

    if model_dir is None:
        model_dir = path.join(here, "data", "models")
    model_dir = process_dir(model_dir)

    device = misc.get_device(device=device)

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        device=device,
    )


if __name__ == "__main__":
    args = parse_input()
    training(args)