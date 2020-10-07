from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.data import Dataset
import numpy as np


def model(img_width=224, img_height=224):
    model_vgg16_conv = VGG16(
        weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3)
    )

    x = Flatten(name="flatten")(model_vgg16_conv.output)
    x = Dense(1024, activation="relu", name="fc1")(x)
    x = Dense(1024, activation="relu", name="fc2")(x)
    x = Dense(2, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=model_vgg16_conv.inputs, outputs=x)

    return model
