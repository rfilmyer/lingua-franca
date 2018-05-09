import os
import argparse
import tempfile

import tensorflow as tf


# # # MODEL SETTINGS
MODEL_DIR = os.path.join(tempfile.gettempdir(), "lingua-franca-model")

parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")
parser.add_argument("--modeldir", default=MODEL_DIR, help="Directory in which to find the model info")

predict_fn = tf.contrib.predictor.from_saved_model(MODEL_DIR)
predictions = predict_fn(
    {"x": [[6.4, 3.2, 4.5, 1.5],
           [5.8, 3.1, 5.0, 1.7]]})
print(predictions['scores'])

# tf.logging.debug("processing NCF data")
# ncf_languages = ["english", "german", "italian"]
# ncf_files = [os.path.join("test-audio", "{0}.wav".format(language)) for language in ncf_languages]
# ncf_data = np.array([randomCrop(create_mfcc(filename)) for filename in ncf_files]).astype(np.float32)
# ncf_labels = np.array([np.where(languages == language) for language in ncf_languages]).flatten()

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