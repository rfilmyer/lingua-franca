# Tensorflow Imports
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.logging.set_verbosity(tf.logging.INFO)

# Sound file stuff
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features.base import mfcc
import random


# Manage our files
import voxforge
import os
import tempfile

# Verbosity flag
import argparse

# # # MODEL SETTINGS
MODEL_DIR = os.path.join(tempfile.gettempdir(), "lingua-franca-model")

parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")
parser.add_argument("--modeldir", default=MODEL_DIR, help="Directory in which to find the model info")

# # # SOUND FILES

MFCC_FILE_NAME = "mfccs.npz"

def create_mfcc(filename: str) -> np.ndarray:
    bitrate, signal = wav.read(filename)
    mfcc_data = mfcc(signal, bitrate, nfft=1200)
    return mfcc_data


NUM_LANGUAGES = 3  # This is a default that should get reset

# # # IMAGE SETTINGS
IMAGE_WIDTH = 13
IMAGE_HEIGHT = 300


def image_is_big_enough(img: np.ndarray, width: int=IMAGE_WIDTH, height: int=IMAGE_HEIGHT) -> bool:
    return img.shape[0] >= height and img.shape[1] >= width


def randomCrop(img: np.ndarray, width: int=IMAGE_WIDTH, height: int=IMAGE_HEIGHT) -> np.ndarray:
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img



# # # MODEL ARCHITECTURE

# Start off:
# 2 Convolutional layers: 5x5, 5x5, ReLU, 2x2/2 pooling
# Fully Connected Layer
# Softmax

# Current Model architecture:
# 6 convolutional layers: 7x7, 5x5, 3x3, with ReLU, 3x3/2 pooling
# One fully connected layer
# Final softmax layer

# Ultimate Model architecture:
# 6 convolutional layers: 7x7, 5x5, 3x3, 3x3, 3x3, 3x3, with ReLU
# # Pooling size is always 3x3 with stride 2
# One fully connected layer
# Final softmax layer

# Input Layer
# we do something custom here
def cnn_model_fn(features, labels, mode) -> tf.estimator.EstimatorSpec:
    input_layer = tf.reshape(features["mfccs"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    tf.logging.debug("Input Layer Shape: %s", input_layer.shape)

    #ROUND1#####################################################################
    # Convolutional/Pooling layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu
    )
    tf.logging.debug("Conv 1 Layer Shape: %s", conv1.shape)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    tf.logging.debug("Pool 1 Layer Shape: %s", pool1.shape)
    # #############################################################################
    # ROUND2#######################################################################
    # Convolutional and Pooling Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    tf.logging.debug("Conv 2 Shape: %s", conv2.shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
    tf.logging.debug("Pool 2 Shape: %s", pool2.shape)
    # #############################################################################
    # ROUND3#####################################################################
    # Convolutional/Pooling layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    tf.logging.debug("Conv 3 Layer Shape: %s", conv3.shape)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=1)
    # ###########################################################################
    # ROUND4#####################################################################
    # Convolutional/Pooling layer 4
    """conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    tf.logging.debug("Conv 4 Layer Shape: %s", conv4.shape)

    
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[4, 4], strides=2)"""
    ############################################################################
    # Fully Connected Layer
    pool3_flat = tf.reshape(pool3, [-1, 1 * 73 * 64])  # These dimensions should match those of the final pooling layer
    tf.logging.debug("Pool 3 Flat Shape: %s", pool3_flat.shape)
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    tf.logging.debug("Dense Shape: %s", dense.shape)
    dropout = tf.layers.dropout(inputs=dense, training=mode == tf.estimator.ModeKeys.TRAIN)
    tf.logging.debug("Dropout Shape: %s", dropout.shape)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=NUM_LANGUAGES)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.debug("Making Predictions...")
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"pred_output_classes": predictions,
                                                                               'probabilities': predictions["probabilities"]})}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Loss Function
    tf.logging.debug("Labels Shape: %s, Logits Shape: %s", labels.shape, logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.logging.debug("Loss Shape: %s", loss.shape)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar("accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.logging.warn("Current mode is %s, expected %s", mode, (tf.estimator.ModeKeys.PREDICT,
                                                              tf.estimator.ModeKeys.TRAIN,
                                                              tf.estimator.ModeKeys.EVAL))


def main(unused_argv):

    # Have we computed MFCCs before?
    if os.path.exists(MFCC_FILE_NAME):
        tf.logging.info("Loading saved images...")
        loaded_data = np.load(MFCC_FILE_NAME)
        data = loaded_data["data"]
        raw_labels = loaded_data["raw_labels"]
    else:
        tf.logging.info("Creating new images...")
        images = []
        raw_labels = []
        for filename, language in voxforge.get_files():
            image = np.zeros([1, 1])
            try:
                image = create_mfcc(filename)
            except ValueError:
                tf.logging.warn("An audio file is messed up: %s", filename)
            if image_is_big_enough(image):
                cropped = randomCrop(image)
                images.append(cropped)
                raw_labels.append(language)
            else:
                tf.logging.debug("Small image: size is {image_shape}, "
                                 "min size is {height}x{width}".format(image_shape=image.shape,
                                                                       width=IMAGE_WIDTH,
                                                                       height=IMAGE_HEIGHT))
        tf.logging.debug("Done converting images.")
        data = np.array(images).astype(np.float32)
        raw_labels = np.array(raw_labels)
        np.savez_compressed(MFCC_FILE_NAME, data=data, raw_labels=raw_labels)

    tf.logging.debug("Data Shape: %s", data.shape)

    language_list = np.unique(raw_labels)
    tf.logging.debug("Languages detected: %s", language_list)
    global NUM_LANGUAGES
    NUM_LANGUAGES = len(language_list)

    labels = np.array([np.where(language_list == language) for language in raw_labels]).flatten()
    tf.logging.debug("Labels Shape: %s", labels.shape)

    train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.10, random_state=42)
    tf.logging.debug("Split Training/Testing data.")

    # Create the Estimator
    tf.logging.info("Model Directory: %s", MODEL_DIR)
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    # Set up logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"mfccs": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=20000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"mfccs": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("Eval Results: %s" % eval_results)

    def serving_input_receiver_fn():
        """Build the serving inputs."""
        # The outer dimension (None) allows us to batch up inputs for
        # efficiency. However, it also means that if we want a prediction
        # for a single instance, we'll need to wrap it in an outer list.
        tf.logging.debug("building input receiver")
        inputs = {"mfccs": tf.placeholder(shape=[None, 1], dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    export_dir = mnist_classifier.export_savedmodel(
        export_dir_base=MODEL_DIR,
        serving_input_receiver_fn=serving_input_receiver_fn)

    print("Model exported to: {0}".format(export_dir))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.v:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    MODEL_DIR = args.modeldir

    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    tf.app.run()
