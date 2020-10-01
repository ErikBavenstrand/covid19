from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from generate import read_tfrecord_files

IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_TRAINING_SAMPLES = 2048
BATCH_SIZE = 32

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

dataset_training = read_tfrecord_files().take(512)

model.fit(dataset_training.batch(BATCH_SIZE).repeat(),
          steps_per_epoch=NUM_TRAINING_SAMPLES // BATCH_SIZE,
          epochs=100)