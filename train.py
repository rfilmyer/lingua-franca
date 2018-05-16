# Tensorflow Imports
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.logging.set_verbosity(tf.logging.INFO)

# Sound file stuff
import numpy as np


# Manage our files
import voxforge
import os
import tempfile
import lingua_franca_config

# Verbosity flag
import argparse

# # # MODEL SETTINGS
MODEL_DIR = os.path.join(tempfile.gettempdir(), "lingua-franca-model")

parser = argparse.ArgumentParser(description="Tensorflow CNN model for language detection")
parser.add_argument("-v", action="store_true", help="Turn on verbose logging")
parser.add_argument("--modeldir", default=MODEL_DIR, help="Directory in which to find the model info")

# # # SOUND FILES

MFCC_FILE_NAME = "mfccs.npz"


NUM_LANGUAGES = 3  # This is a default that should get reset



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
    tf.logging.debug("Feature Shape: %s", features["mfccs"].shape)
    input_layer = tf.reshape(features["mfccs"], [-1,
                                                 lingua_franca_config.num_frames,
                                                 lingua_franca_config.num_cepstra,
                                                 1])
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

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
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
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
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
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    tf.logging.debug("Conv 4 Layer Shape: %s", conv4.shape)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=1)
    tf.logging.debug("Pool 4 Shape: %s", pool4.shape)
    ############################################################################
    # Fully Connected Layer

    # framelengths to final size:
    # 300 :: 73, 250 :: 60, 400 :: 98
    # 13 :: 1, 24 :: 5, 26 ::5, 18 :: 3, 17:: 3

    # final_frame_length = np.floor_divide(lingua_franca_config.num_frames, 4) - 2
    # final_frame_length = conv4.shape[]
    # final_cepstra_height = max(conv4.shape[3] - 1, 1) # screw it, I can't figure it out

    # These dimensions should match those of the final pooling layer
    # For 13x300 this should be (batch_size, 73, 1, 64)
    # pool4_flat = tf.reshape(pool4, [-1, final_cepstra_height * final_frame_length * 64])

    # Suggestion: Just match it automatically
    pool4_flat = tf.reshape(pool4, [-1, np.product(pool4.shape[1:])])
    tf.logging.debug("Pool 4 Flat Shape: %s", pool4_flat.shape)
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
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
        tf.logging.debug("Probabilities: %s", predictions["probabilities"])
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"classes": predictions["classes"],
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


def serving_input_receiver_fn():
    """Build the serving inputs."""
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    tf.logging.debug("building input receiver")
    inputs = {"mfccs": tf.placeholder(shape=[None, lingua_franca_config.num_frames, lingua_franca_config.num_cepstra],
                                      dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def regenerate_images() -> tuple:
    """
    Create MFCCs from Voxforge WAV files.

    Returns a tuple (data, raw_labels), with 2 ndarrays
    """
    tf.logging.info("Creating new images...")
    images = []
    raw_labels = []
    for filename, language in voxforge.get_files()[:200]:
        image = np.zeros([1, 1])
        try:
            image = voxforge.create_mfcc(filename)
        except ValueError:
            tf.logging.warn("An audio file is messed up: %s", filename)
        if voxforge.image_is_big_enough(image):
            cropped = voxforge.randomCrop(image)
        else:
            padded = np.zeros((lingua_franca_config.num_frames, lingua_franca_config.num_cepstra))
            frames, cepstra = image.shape
            padded[0:frames, 0:cepstra] = image

        images.append(cropped)
        raw_labels.append(language)


    tf.logging.info("Created %d MFCCs from %d images.", len(images), len(voxforge.get_files()))
    data = np.array(images).astype(np.float32)
    raw_labels = np.array(raw_labels)
    np.savez_compressed(MFCC_FILE_NAME, data=data, raw_labels=raw_labels)
    return data, raw_labels


def main(unused_argv):
    # Have we computed MFCCs before?
    if os.path.exists(MFCC_FILE_NAME):
        tf.logging.info("Loading saved images...")
        loaded_data = np.load(MFCC_FILE_NAME)
        if isinstance(loaded_data["data"], np.ndarray) and \
                len(loaded_data["data"]) > 0 and \
                loaded_data["data"][0].shape == (lingua_franca_config.num_frames, lingua_franca_config.num_cepstra):
            data = loaded_data["data"]
            raw_labels = loaded_data["raw_labels"]
        else:
            tf.logging.warn("Precomputed MFCCs are of wrong size, expected %s, got %s. Will have to recompute.",
                            (lingua_franca_config.num_frames, lingua_franca_config.num_cepstra),
                            loaded_data["data"].shape)
            data, raw_labels = regenerate_images()


    else:
        data, raw_labels = regenerate_images()


    tf.logging.debug("Data Shape: %s", data.shape)

    language_list = np.sort(np.unique(raw_labels))
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
        batch_size=lingua_franca_config.batch_size,
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

    export_dir = mnist_classifier.export_savedmodel(
        export_dir_base=MODEL_DIR,
        serving_input_receiver_fn=serving_input_receiver_fn)

    print("Model exported to: {0}".format(export_dir))

    with open(os.path.join(export_dir, b"languages.csv"), 'w') as language_csv:
        for language in language_list:
            language_csv.write(language)
            language_csv.write("\n")



if __name__ == "__main__":
    args = parser.parse_args()

    if args.v:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    MODEL_DIR = args.modeldir

    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    tf.app.run()
