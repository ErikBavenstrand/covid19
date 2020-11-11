import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model


def model(img_width=224, img_height=224):
    model_resnet50_conv = ResNet50V2(
        include_top=False, input_shape=(img_width, img_height, 3)
    )
    for layer in model_resnet50_conv.layers[:154]:
        layer.trainable = False

    x = GlobalAveragePooling2D(name="GAP")(model_resnet50_conv.output)
    x = Dense(2, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=model_resnet50_conv.inputs, outputs=x)

    return model
