import os
import argparse
import tempfile

import tensorflow as tf
import numpy as np

import pandas as pd

from voxforge import create_mfcc, randomCrop
import lingua_franca_config

# # # MODEL SETTINGS
MODEL_DIR = os.path.join(tempfile.gettempdir(), "lingua-franca-model")

parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")
parser.add_argument("--modeldir", default=MODEL_DIR, help="Directory in which to find the model info")

def load_language_list(filename = lingua_franca_config.mfccs_file_name):
    """
    Builds the list of classified languages from the original data
    """
    tf.logging.debug("loading language list")
    loaded_data = np.load(filename)
    raw_labels = loaded_data["raw_labels"]
    language_list = np.sort(np.unique(raw_labels))
    return language_list


def load_ncf_files(directory="test-audio"):
    """
    Converts test files into MFCCs and pulls their metadata
    """
    tf.logging.debug("processing NCF data")
    ncf_files = pd.read_csv(os.path.join(directory, "files.csv"))

    mfcc_list = []
    for filename in ncf_files["filename"]:
        filename = os.path.join(directory, filename)
        mfcc = create_mfcc(filename)
        cropped_mfcc = randomCrop(mfcc)
        mfcc_list.append(cropped_mfcc)

    mfcc_array = np.array(mfcc_list, np.float32)
    return ncf_files, mfcc_array


if __name__ == "__main__":

    args = parser.parse_args()
    MODEL_DIR = args.modeldir
    if args.v:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    ncf_files, ncf_mfccs = load_ncf_files()
    predictions = ncf_files[["filename", "language"]].copy()

    if os.path.exists(os.path.join(MODEL_DIR, "languages.csv")):
        language_list = np.genfromtxt(os.path.join(MODEL_DIR, "languages.csv"), dtype='str')
    else:
        language_list = load_language_list()

    predict_fn = tf.contrib.predictor.from_saved_model(MODEL_DIR)
    raw_predictions = predict_fn(
        {"mfccs": ncf_mfccs})

    predictions["predicted"] = [language_list[prediction] for prediction in raw_predictions["classes"]]
    predictions["pct_confidence"] = np.amax(raw_predictions["probabilities"], axis=1)
    print("PREDICTIONS:")
    print(predictions)
