import os
import onnx
import click
import logging
import ultralytics
import mlflow
import mlflow.pyfunc
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ultralytics import settings


@click.command()
@click.argument('data_yaml', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['yolov8', 'yolov9']), default='yolov8', help='YOLO model version to use.')
@click.option('--epochs', type=int, default=3, help='Number of epochs to train for.')
@click.option('--batch-size', type=int, default=4, help='Batch size for training.')
@click.option('--img-size', type=int, default=640, help='Image size for training.')
@click.option('--project', type=str, default='runs/train', help='Save directory for training results.')
@click.option('--name', type=str, default='exp', help='Experiment name.')
@click.option('--experiment-name', type=str, default='YOLO Training', help='MLflow experiment name.')
@click.option('--run-name', type=str, default=None, help='MLflow run name.')
def main(data_yaml, model, epochs, batch_size, img_size, project, name, experiment_name, run_name):
    """
    Trains a YOLOv8 or YOLOv9 model and logs to MLflow.
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting model training')

    # Set up MLflow tracking server
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN"] = run_name
    settings.update({"mlflow": True})

    # Model selection
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

    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment['experiment_id']
    run_id = mlflow.search_runs([experiment_id], filter_string=f"run_name='{run_name}'")['run_id'].loc[0]
    mlflow.register_model(f"runs:/{run_id}/weights/best.pt", f'{model}_{experiment_name}_{run_name}_{img_size}.pt')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
