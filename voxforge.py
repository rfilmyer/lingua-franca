import os
import random

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features.base import mfcc

import lingua_franca_config

from typing import List, Tuple, Union
FilePath = Union[str, bytes]



def get_files(voxforge_directory: FilePath = "voxforge") -> List[Tuple[FilePath, str]]:
    """
    Discovers audio files in the voxforge directory and returns their paths and languages.

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


# # # IMAGE SETTINGS
IMAGE_WIDTH = lingua_franca_config.num_cepstra
IMAGE_HEIGHT = lingua_franca_config.num_frames


def image_is_big_enough(img: np.ndarray,
                        width: int=lingua_franca_config.num_cepstra,
                        height: int=lingua_franca_config.num_frames) -> bool:
    return img.shape[0] >= height and img.shape[1] >= width


def randomCrop(img: np.ndarray,
               width: int=lingua_franca_config.num_cepstra,
               height: int=lingua_franca_config.num_frames) -> np.ndarray:
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img

def create_mfcc(filename: str) -> np.ndarray:
    bitrate, signal = wav.read(filename)
    mfcc_data = mfcc(signal, bitrate, numcep=lingua_franca_config.num_cepstra, nfft=1200)
    return mfcc_data

if __name__ == "__main__":

    for entry in get_files():
        print(entry)


