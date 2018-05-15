import os
import argparse
import tempfile

import tensorflow as tf
import numpy as np

import pandas as pd

from voxforge import create_mfcc, randomCrop

# # # MODEL SETTINGS
MODEL_DIR = os.path.join(tempfile.gettempdir(), "lingua-franca-model")

parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")
parser.add_argument("--modeldir", default=MODEL_DIR, help="Directory in which to find the model info")

# To build the language list, we need to load the master file of MFCCs.
MFCC_FILE_NAME = "mfccs.npz"
loaded_data = np.load(MFCC_FILE_NAME)
raw_labels = loaded_data["raw_labels"]
language_list = np.sort(np.unique(raw_labels))


tf.logging.debug("processing NCF data")
ncf_languages = ["english", "german", "italian"]
ncf_files = [os.path.join("test-audio", "{0}.wav".format(language)) for language in ncf_languages]
ncf_data = np.array([randomCrop(create_mfcc(filename)) for filename in ncf_files]).astype(np.float32)
ncf_labels = np.array([np.where(language_list == language) for language in ncf_languages]).flatten()

# tf.logging.debug("Evaluating our data")
# eval_ncf_audio_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": ncf_data},
#     y=ncf_labels,
#     num_epochs=1,
#     shuffle=False
# )
#
# ncf_results = mnist_classifier.evaluate(input_fn=eval_ncf_audio_fn)
# print("NCF Results: %s" % ncf_results)

if __name__ == "__main__":

    predictions = pd.DataFrame(ncf_files, columns=["filename"])
    predictions["expected"] = ncf_languages

    args = parser.parse_args()
    MODEL_DIR = args.modeldir
    predict_fn = tf.contrib.predictor.from_saved_model(MODEL_DIR)
    raw_predictions = predict_fn(
        {"mfccs": ncf_data})

    predictions["predicted"] = [language_list[prediction] for prediction in raw_predictions["pred_output_classes"]]
    predictions["pct_confidence"] = np.amax(raw_predictions["probabilities"], axis=1)
    print("PREDICTIONS:")
    print(predictions)