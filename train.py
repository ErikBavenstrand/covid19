import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

import wandb
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from wandb.keras import WandbCallback
from generate import read_tfrecord_files, get_tfrecord_sample_count
from Models import simple_cnn, vgg16
from Utils import grad_cam, make_dir


def get_callbacks(config):
    callbacks = []

    if config.save_model:
        print('Saving model as ' + config.save_model)
        filepath = './Saved Models'
        make_dir(filepath)
        callbacks.append(
            ModelCheckpoint(filepath=filepath + '/' + config.save_model,
                            save_best_only=True,
                            save_weights_only=False,
                            monitor='val_accuracy',
                            mode='max',
                            verbose=1))
    return callbacks


def train(model, train_dataset, validation_dataset, config):
    callbacks = get_callbacks(config)
    if not config.no_wandb:
        callbacks.append([WandbCallback(save_model=False)])

    model.fit(
        train_dataset.batch(config.batch_size).repeat(),
        steps_per_epoch=config.train_sample_count // config.batch_size,
        validation_data=validation_dataset.batch(config.batch_size).repeat(),
        validation_steps=config.validation_sample_count // config.batch_size,
        callbacks=callbacks,
        epochs=config.epochs)

    return model


def test():
    pass


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--train-split',
                        type=float,
                        default=0.9,
                        metavar='f',
                        help='train/validation split (default: 0.9)')
    parser.add_argument('--model',
                        type=str,
                        default='simple_cnn',
                        metavar='MODEL',
                        help='model name (default: simple_cnn)')
    parser.add_argument('--save-model',
                        type=str,
                        metavar='FILENAME',
                        help='filename of model weights')
    parser.add_argument('--no-wandb',
                        action='store_true',
                        help='do not send the results to wandb')

    args = parser.parse_args()
    config = args

    if not config.no_wandb:
        wandb.init(config=args,
                   project='covid19',
                   entity='erikbavenstrand',
                   save_code=False)
        config = wandb.config

    config.sample_count = get_tfrecord_sample_count()
    config.train_sample_count = int(config.train_split * config.sample_count)
    config.validation_sample_count = int(
        (1 - config.train_split) * config.sample_count)

    dataset = read_tfrecord_files()
    train_dataset = dataset.take(config.train_sample_count)
    validation_dataset = dataset.skip(config.train_sample_count).take(
        config.validation_sample_count)

    if config.model == 'simple_cnn':
        model = simple_cnn.model()
    elif config.model == 'vgg16':
        model = vgg16.model()
    else:
        raise ValueError("Model does not exist. Check ./Models")
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model = train(model, train_dataset, validation_dataset, config)
    test()


if __name__ == "__main__":
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU")
    else:
        print("Using CPU")
    main()