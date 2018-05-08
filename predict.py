"""Make one-off predictions from model trained by train2.py.

Author: Joe
Date: 05/06/2018
"""

import tensorflow as tf
import os
import numpy as np
from python_speech_features.base import mfcc
import random

from voxforge import create_mfcc, image_is_big_enough, randomCrop


def main():
    inFile = input("Path to language sample?")
    sample = create_mfcc(inFile)
    if image_is_big_enough(sample):
        sample = randomCrop(sample)



    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
        predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
        model_input = tf.train.Example(features=tf.train.Features(feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set))})) 
        # model_input = model_input.SerializeToString()
        output_dict = predictor({"predictor_inputs": [sample]})
        y_predicted = output_dict["pred_output_classes"][0]
