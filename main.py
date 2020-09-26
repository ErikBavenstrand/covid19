# prevent tf debug logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import wandb
from wandb import magic

wandb.init(magic=True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255
x_validation, y_validation = x_train[55000:], y_train[55000:]
x_train, y_train = x_train[:55000], y_train[:55000]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x=x_train,
          y=y_train,
          validation_data=(x_validation, y_validation),
          epochs=5)

model.evaluate(x=x_test, y=y_test, verbose=2)
