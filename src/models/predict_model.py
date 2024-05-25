import click
import logging
from pathlib import Path
from ultralytics import YOLO
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('--model', type=click.Path(exists=True), default='yolov8', help='YOLO model version to use.')
def main(model: str = "path_to_trained_model.pt", input_filepath: str = "path_to_images"):
    """
    Makes predictions using a trained YOLO model.
    """
    yolo = YOLO(model)
    results = yolo.predict(input_filepath, save=True)
    logging.info(results)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
