import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

import wandb
from wandb.keras import WandbCallback
from generate import read_tfrecord_files
from Models import simple_cnn


def train(model, train_dataset, validation_dataset, config):
    train_size = int(0.85 * 8000)
    val_size = int(0.15 * 8000)
    model.fit(train_dataset.batch(config.batch_size).repeat(),
              steps_per_epoch=train_size // config.batch_size,
              validation_data=validation_dataset.batch(
                  config.batch_size).repeat(),
              validation_steps=val_size // config.batch_size,
              callbacks=[WandbCallback()],
              epochs=config.epochs)


def test():
    pass


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no_wandb',
                        action='store_true',
                        help='do not send the results to wandb')

    args = parser.parse_args()
    config = args

    if not config.no_wandb:
        wandb.init(config=args, project='covid19', entity='erikbavenstrand')
        config = wandb.config

    NUM_TRAINING_SAMPLES = 8000
    BATCH_SIZE = 64
    train_size = int(0.85 * NUM_TRAINING_SAMPLES)
    val_size = int(0.15 * NUM_TRAINING_SAMPLES)

    dataset = read_tfrecord_files()
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size).take(val_size)

    model = simple_cnn.model()

    train(model, train_dataset, validation_dataset, config)


if __name__ == "__main__":
    main()