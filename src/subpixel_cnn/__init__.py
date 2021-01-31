"""
Code modified from https://github.com/keras-team/keras-io/blob/master/examples/vision/super_resolution_sub_pixel.py
An implementation of W. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network, Computer Vision and Pattern Recognition*, **2016**
"""
from typing import Tuple

import keras
import tensorflow
import numpy


def initialize_subpixel_cnn(upscale_factor=2, channels=1) -> keras.models.Model:
    """
    Initializes a subpixel CNN model
    :param upscale_factor: Upscale factor of model
    :param channels: Number of color channels in image
    :return: Model created using Keras functional API
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = keras.layers.Conv2D(64, 5, **conv_args)(inputs)
    x = keras.layers.Conv2D(64, 3, **conv_args)(x)
    x = keras.layers.Conv2D(32, 3, **conv_args)(x)
    x = keras.layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tensorflow.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)


def load_subpixel_cnn(filename) -> Tuple[int, int, keras.models.Model]:
    """
    Loads subpixel cnn model from tensorflow or keras h5 model file
    :param filename: Path to file to load from
    :return: Tuple containing # of channels, upscale factor, and model object
    """
    model = keras.models.load_model(filename)
    # Infer upscale factor, channels from model structure
    channels = model.input_shape[-1]
    upscale = int(numpy.round(numpy.log(model.layers[4].filters / channels)/numpy.log(2)))
    return channels, upscale, model
