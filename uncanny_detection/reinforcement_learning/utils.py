import torch
from imgaug import augmenters as iaa
import os
import numpy as np
from transformers.image_utils import load_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_THRESHOLDS = [0.4, 0.3]

IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'uncanny')
NOT_UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'not_uncanny')


def load_images(folder, is_uncanny):
    images = []

    flip_augmenter = iaa.Fliplr(1.0)  # Flip all images horizontally
    contrast_augmenter = iaa.LinearContrast((0.75, 1.5))  # Adjust contrast

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        print(f"Loading {filepath}")
        image = load_image(filepath)
        if image is not None:
            # np_image = np.array(image)

            # Original image
            images.append({'image': image, 'is_uncanny': is_uncanny})

            # Flipped image
            # flipped_np = flip_augmenter(image=np_image)
            # flipped_image = Image.fromarray(flipped_np)
            # images.append({'image': flipped_image, 'is_uncanny': is_uncanny})

            # Contrast-adjusted image
            # contrast_np = contrast_augmenter(image=np_image)
            # contrast_image = Image.fromarray(contrast_np)
            # images.append({'image': contrast_image, 'is_uncanny': is_uncanny})

            # Both flipped and contrast-adjusted image
            # flipped_contrast_np = contrast_augmenter(image=flipped_np)
            # flipped_contrast_image = Image.fromarray(flipped_contrast_np)
            # images.append({'image': flipped_contrast_image, 'is_uncanny': is_uncanny})

    return images

def calculate_confusion_matrix(true_positive, false_positive, true_negative, false_negative):
    accuracy = ((true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)) if (true_positive + false_positive + true_negative + false_negative) > 0 else 0
    precision = (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0
    recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0
    return accuracy, precision, recall