"""Preprocess audio files.

Take files from the folder voxforge, create mfccs and output results to csv with one mfcc per row.
"""
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# Sound file stuff
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features.base import mfcc
import random


# Manage our files
import os
import pandas as pd
import pickle

from typing import List, Tuple, Union
FilePath = Union[str, bytes]

# # # IMAGE SETTINGS
IMAGE_WIDTH = 13
IMAGE_HEIGHT = 300

def get_files(voxforge_directory: FilePath = "voxforge") -> List[Tuple[FilePath, str]]:
    """
    Discover audio files in the voxforge directory and returns their paths and languages.

    :return: A list of tuples containing the file path of a soundfile and its language.
    """
    file_list = []

    language_folders = next(os.walk(voxforge_directory))[1]
    for language_folder in language_folders:
        language_folder_path = os.path.join(voxforge_directory, language_folder)

        files_in_language_folder = next(os.walk(language_folder_path))[2]
        for soundfile in files_in_language_folder:
            soundfile_path = os.path.join(language_folder_path, soundfile)

            file_list.append((soundfile_path, language_folder))

    return file_list


def create_mfcc(filename: str) -> np.ndarray:
    """Create MFCC array from WAV file."""
    bitrate, signal = wav.read(filename)
    mfcc_data = mfcc(signal, bitrate, nfft=1200)
    return mfcc_data


def image_is_big_enough(img: np.ndarray, width: int=IMAGE_WIDTH, height: int=IMAGE_HEIGHT) -> bool:
    return img.shape[0] >= height and img.shape[1] >= width


def randomCrop(img: np.ndarray, width: int=IMAGE_WIDTH, height: int=IMAGE_HEIGHT) -> np.ndarray:
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img



file_list = get_files()
languages = np.unique([entry[1] for entry in file_list])
global NUM_LANGUAGES
NUM_LANGUAGES = len(languages)

tf.logging.debug("Creating images...")
images = []
raw_labels = []
for filename, language in file_list:
    try:
        image = create_mfcc(filename)
    except ValueError as e:
        tf.logging.warn("An audio file is messed up: %s", filename)
    if image_is_big_enough(image):
        cropped = randomCrop(image)
        images.append(cropped.tolist())
        raw_labels.append(language)
    else:
        tf.logging.debug("Small image: size is {image_shape}, min size is {height}x{width}".format(image_shape=image.shape,
                                                                                                   width=IMAGE_WIDTH,
                                                                                                   height=IMAGE_HEIGHT))

with open('mfccs.pkl', 'wb') as mfcc_outFile:
    pickle.dump(images, mfcc_outFile)

with open('raw_label.pkl', 'wb') as label_outFile:
    pickle.dump(raw_labels, label_outFile)

