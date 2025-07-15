import torch
from imgaug import augmenters as iaa
import os
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_THRESHOLDS = [0.4, 0.3]

IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'uncanny')
NOT_UNCANNY_FOLDER = os.path.join(IMAGE_FOLDER, 'not_uncanny')


def load_images(folder, is_uncanny):
    images = []

    # Define individual augmenters
    flip_augmenter = iaa.Fliplr(1.0)  # Flip all images horizontally
    contrast_augmenter = iaa.LinearContrast((0.75, 1.5))  # Adjust contrast

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        print(f"Loading {filepath}")
        image = cv2.imread(filepath)
        if image is not None:
            # Original image
            images.append({'image': image, 'is_uncanny': is_uncanny})

            # Flipped image
            flipped_image = flip_augmenter(image=image)
            images.append({'image': flipped_image, 'is_uncanny': is_uncanny})

            # Contrast-adjusted image
            contrast_image = contrast_augmenter(image=image)
            images.append({'image': contrast_image, 'is_uncanny': is_uncanny})

            # Both flipped and contrast-adjusted image
            flipped_contrast_image = contrast_augmenter(image=flipped_image)
            images.append({'image': flipped_contrast_image,
                          'is_uncanny': is_uncanny})

    return images
