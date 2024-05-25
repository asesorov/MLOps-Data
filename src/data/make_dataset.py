import click
import logging
import yaml
import shutil
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def validate_dataset_structure(input_filepath):
    subsets = ['train', 'valid', 'test']
    for subset in subsets:
        images_path = input_filepath / subset / 'images'
        labels_path = input_filepath / subset / 'labels'

        if not images_path.is_dir() or not labels_path.is_dir():
            logging.error(f"Missing '{subset}/images' or '{subset}/labels' directories.")
            return False

        images = list(images_path.glob('*.jpg'))
        labels = list(labels_path.glob('*.txt'))

        if len(images) < len(labels):
            logging.error(f"There are more label files than images in {subset}.")
            return False

    return True


def check_yaml_structure(input_filepath):
    yaml_path = input_filepath / 'data.yaml'
    if not yaml_path.exists():
        logging.error("Missing 'data.yaml' file.")
        return False

    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    required_keys = {'train', 'val', 'test', 'nc', 'names'}
    if not required_keys.issubset(data.keys()):
        logging.error("YAML file missing required keys.")
        return False

    return True


def copy_dataset(input_filepath, output_filepath):
    for path in input_filepath.rglob('*'):
        if path.is_file():
            relative_path = path.relative_to(input_filepath)
            dest_path = output_filepath / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest_path)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, file_okay=False))
@click.argument('output_filepath', type=click.Path())
@click.option('--augment_percent', default=20, help='Percentage of augmented images to add, e.g., 20 means 20%.')
def main(input_filepath, output_filepath, augment_percent):
    """
    Validates and augments the dataset based on the specified percentage.
    """
    logger = logging.getLogger(__name__)
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    if not validate_dataset_structure(input_filepath):
        raise ValueError(f'Dataset structure is invalid: {input_filepath}.')

    if not check_yaml_structure(input_filepath):
        raise ValueError(f'data.yaml structure is invalid: {input_filepath}.')

    copy_dataset(input_filepath, output_filepath)

    logger.info('Applying augmentations to the dataset')

    augmentations = A.Compose([
        A.RandomBrightnessContrast(p=0.4),
        A.RandomGamma(p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        A.RandomShadow(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for subset in ['train', 'valid', 'test']:
        images_path = output_filepath / subset / 'images'
        labels_path = output_filepath / subset / 'labels'

        images = list(images_path.glob('*.jpg'))
        num_augmented = int(len(images) * (augment_percent / 100))

        for _ in tqdm(range(num_augmented), desc=f"Augmenting {subset} images"):
            i = np.random.randint(0, len(images))
            image_path = images[i]
            label_path = labels_path / f"{image_path.stem}.txt"

            if not label_path.exists():
                logger.warning(f"Label file not found for {image_path.name}, skipping.")
                continue

            image = cv2.imread(str(image_path))
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]

            if not labels:
                logging.warning(f'No labels found for {subset}')
                continue

            class_labels = [int(label[0]) for label in labels]
            bboxes = [[float(x) for x in label[1:]] for label in labels]

            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = augmented['image']
            transformed_bboxes = augmented['bboxes']
            transformed_labels = augmented['class_labels']

            new_image_path = images_path / f"{image_path.stem}_aug_{_}.jpg"
            new_label_path = labels_path / f"{image_path.stem}_aug_{_}.txt"

            cv2.imwrite(str(new_image_path), transformed_image)
            with open(new_label_path, 'w') as f:
                for label, bbox in zip(transformed_labels, transformed_bboxes):
                    # Ensure bbox is a list (not a tuple) and concatenate correctly
                    bbox_str = ' '.join(map(str, [label] + list(bbox)))
                    f.write(bbox_str + '\n')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
