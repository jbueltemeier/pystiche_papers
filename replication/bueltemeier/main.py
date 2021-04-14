import csv
import os
from argparse import Namespace
from datetime import datetime
from os import path

import pystiche_papers.bueltemeier as paper
from pystiche import image, misc
from pystiche_papers import utils
from pystiche_papers.bueltemeier._utils import hyper_parameters as _hyper_parameters


def training(args):
    contents = ("karya", "004", "04", "bueltemeier")
    styles = ("UHD20", "DM100", "MAD20", "Specimen0")

    dataset = paper.dataset(path.join(args.dataset_dir, "content"),)
    image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda")
    )

    images = paper.images(args.image_source_dir)

    for style in styles:
        style_image = images[style].read(device=args.device, size=512)

        transformer = paper.training(image_loader, style_image)

        model_name = f"bueltemeier_2021__{style}"
        file = utils.save_state_dict(transformer, model_name, root=args.model_dir)

        hyper_parameters = _hyper_parameters()

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H_%M_%S")

        dict_data = {
            "filename": file,
            "mode": hyper_parameters.loss.mode,
            "regularization": hyper_parameters.regularization.mode,
            "regularization_score_weight": hyper_parameters.regularization.score_weight,
            "transformer_type": hyper_parameters.transformer.type,
            "transformer_levels": hyper_parameters.transformer.levels,
            "num_batches": hyper_parameters.batch_sampler.num_batches,
            "batch_size": hyper_parameters.batch_sampler.batch_size,
            "image_size": hyper_parameters.content_transform.image_size,
            "gram_score_weight": hyper_parameters.gram_style_loss.score_weight,
            "gram_layers": str(len(hyper_parameters.gram_style_loss.layers)),
            "mrf_score_weight": hyper_parameters.mrf_style_loss.score_weight,
            "time": dt_string,
        }
        data_columns = dict_data.keys()
        with open("model_data.csv", mode="a") as data_file:
            writer = csv.DictWriter(data_file, fieldnames=data_columns)
            writer.writerow(dict_data)

        for content in contents:
            content_image = images[content].read(device=args.device)
            output_image = paper.stylization(content_image, transformer)

            output_name = f"{style}_{content}_{dt_string}"
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
