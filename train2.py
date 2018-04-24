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

# Verbosity flag
import argparse


parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")

# # # SOUND FILES
def create_mfcc(filename: str) -> np.ndarray:
    bitrate, signal = wav.read(filename)
    mfcc_data = mfcc(signal, bitrate, nfft=1200)
    return mfcc_data

# # # IMAGE SETTINGS
IMAGE_WIDTH = 13
IMAGE_HEIGHT = 300

def randomCrop(img: np.ndarray, width: int=IMAGE_WIDTH, height: int=IMAGE_HEIGHT) -> np.ndarray:
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width]
    return img

global NUM_LANGUAGES
NUM_LANGUAGES = 3  # This is a default that should get reset



# # # MODEL ARCHITECTURE

# Start off:
# 2 Convolutional layers: 5x5, 5x5, ReLU, 2x2/2 pooling
# Fully Connected Layer
# Softmax


# Ultimate Model architecture:
# 6 convolutional layers: 7x7, 5x5, 3x3, 3x3, 3x3, 3x3, with ReLU
# # Pooling size is always 3x3 with stride 2
# One fully connected layer
# Final softmax layer

# Input Layer
# we do something custom here
def cnn_model_fn(features, labels, mode) -> tf.estimator.EstimatorSpec:
    input_layer = tf.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    print("Input Layer Shape: ", input_layer.shape)

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
    ##############################################################################
    #ROUND2#######################################################################
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
    ##############################################################################
    #ROUND3#####################################################################
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
    ############################################################################
    #ROUND4#####################################################################
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
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss Function
    tf.logging.debug("Labels Shape: %s", labels.shape, "Logits Shape:", logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.logging.debug("Loss Shape: %s", loss.shape)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    file_list = voxforge.get_files()
    languages = np.unique([entry[1] for entry in file_list])
    global NUM_LANGUAGES
    NUM_LANGUAGES = len(languages)

    tf.logging.debug("Creating images...")
    data = np.array([randomCrop(create_mfcc(filename)) for filename, language in file_list]).astype(np.float32)
    tf.logging.debug("Data Shape: %s", data.shape)
    labels = np.array([np.where(languages == language) for filename, language in file_list]).flatten()
    tf.logging.debug("Labels Shape: %s", labels.shape)
    train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=0.10, random_state=42)
    tf.logging.debug("Split Training/Testing data.")

    # tf.logging.debug("processing NCF data")
    # ncf_languages = ["english", "german", "italian"]
    # ncf_files = [os.path.join("test-audio", "{0}.wav".format(language)) for language in ncf_languages]
    # ncf_data = np.array([randomCrop(create_mfcc(filename)) for filename in ncf_files]).astype(np.float32)
    # ncf_labels = np.array([np.where(languages == language) for language in ncf_languages]).flatten()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/lingua-franca-model")

    # Set up logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # tf.logging.debug("Evaluating our data")
    # eval_ncf_audio_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": ncf_data},
    #     y=ncf_labels,
    #     num_epochs=1,
    #     shuffle=False
    )

    # ncf_results = mnist_classifier.evaluate(input_fn=eval_ncf_audio_fn)
    # print(ncf_results)





if __name__ == "__main__":
    args = parser.parse_args()

    if args.v:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    tf.app.run()
