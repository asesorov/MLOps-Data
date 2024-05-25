import click
import logging
import ultralytics
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('data_yaml', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['yolov8', 'yolov9']), default='yolov8', help='YOLO model version to use.')
@click.option('--epochs', type=int, default=3, help='Number of epochs to train for.')
@click.option('--batch-size', type=int, default=16, help='Batch size for training.')
@click.option('--img-size', type=int, default=640, help='Image size for training.')
@click.option('--project', type=str, default='runs/train', help='Save directory for training results.')
@click.option('--name', type=str, default='exp', help='Experiment name.')
def main(data_yaml, model, epochs, batch_size, img_size, project, name):
    """
    Trains a YOLOv8 or YOLOv9 model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting model training')

    if model == 'yolov8':
        model_type = ultralytics.YOLO('yolov8n.pt')
    elif model == 'yolov9':
        model_type = ultralytics.YOLO('yolov9c.pt')
    else:
        raise AttributeError(f'Unknown model type: {model_type}')

    model_type.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
