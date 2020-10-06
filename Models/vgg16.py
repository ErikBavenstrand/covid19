from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.data import Dataset
import numpy as np


def model(img_width=224, img_height=224):
    input_layer = Input(shape=(img_width, img_height, 3))
    model_vgg16_conv = vgg16.VGG16(weights='imagenet',
                                   include_top=False)(input_layer)

    #Add the fully-connected layers
    x = Flatten(name='flatten')(model_vgg16_conv)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    #Create your own model
    my_model = Model(inputs=input_layer, outputs=x)
    my_model.layers[1].trainable = False

    my_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    return my_model