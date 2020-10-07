from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.backend import mean
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.data import Dataset
import numpy as np


def global_average_pooling(x):
    return mean(x, axis=(1, 2))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def model(img_width=224, img_height=224):
    model_vgg16_conv = VGG16(
        weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
    )
    model_vgg16_conv.trainable = False
    x = Lambda(global_average_pooling)(model_vgg16_conv.layers[-2].output)
    x = Dense(2, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=model_vgg16_conv.inputs, outputs=x)

    return model
