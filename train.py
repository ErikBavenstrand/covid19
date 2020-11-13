import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from Utils import make_dir
from Models import simple_cnn, vgg16, vgg16cam, resnet50
from generate import read_tfrecord_files, get_tfrecord_sample_count, make_generator
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import wandb
import argparse
import glob


def get_callbacks(config):
    callbacks = []

    if config.save_model:
        print("Saving model as " + config.save_model)
        filepath = "./Saved Models"
        make_dir(filepath)
        callbacks.append(
            ModelCheckpoint(
                filepath=filepath + "/" + config.save_model,
                save_best_only=True,
                save_weights_only=False,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
                save_freq="epoch",
            )
        )
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
        epochs=config.epochs,
        class_weight={0: 13.49806576, 1: 0.51923363},
    )

    return model


def test_confidence(model, dataset, config):
    # TODO: Refactor import/figure output
    import matplotlib.pyplot as plt

    predictions = model.predict(dataset.batch(config.batch_size))

    # TODO: Move constants to config
    LIMIT = 8
    DISP_ROWS = 2

    # TODO: Assert correct guess by model, right now we're just taking the most
    # likely guesses whether correct or not.
    covid_predictions = predictions[:, 0]
    ordered_predictions = covid_predictions.argsort()
    most_unlikely = ordered_predictions[:LIMIT]
    most_likely_predictions = ordered_predictions[-LIMIT:]

    imgndx = 1
    rows = DISP_ROWS
    cols = LIMIT // DISP_ROWS
    # Can't index into dataset/numpy iterator as far as I know
    for ndx, (record, label) in enumerate(dataset.as_numpy_iterator()):
        if ndx in most_likely_predictions:
            plt.subplot(rows, cols, imgndx)
            plt.imshow(record)
            imgndx += 1
    plt.show()


def test():
    pass


def test_eval(model, dataset, config):
    return model.evaluate(
        dataset.batch(config.batch_size).repeat(),
        steps=config.test_sample_count // config.batch_size,
    )


def test_pred(model, dataset, config):
    return model.predict(
        dataset.batch(config.batch_size).repeat(),
        steps=config.validation_sample_count // config.batch_size,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        metavar="MODEL",
        help="model name (default: resnet50)",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="./data_covidx/",
        metavar="PATH",
        help="path to train set (default: ./data_covidx/)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./data_valencia/",
        metavar="PATH",
        help="path to test set (default: ./data_valencia/)",
    )
    parser.add_argument(
        "--save-model", type=str, metavar="FILENAME", help="filename of model weights"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="do not send the results to wandb"
    )

    args = parser.parse_args()
    config = args

    if not config.no_wandb:
        wandb.init(
            config=args, project="covid19", entity="erikbavenstrand", save_code=False
        )
        config = wandb.config

    if config.model == "simple_cnn":
        model = simple_cnn.model()
    elif config.model == "vgg16":
        model = vgg16.model()
    elif config.model == "vgg16cam":
        model = vgg16cam.model()
    elif config.model == "resnet50":
        model = resnet50.model()
    else:
        raise ValueError("Model does not exist. Check ./Models")

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    train_dataset = tf.data.Dataset.from_generator(
        lambda: make_generator(args.train_path + "train"),
        (tf.float32, tf.float32),
        (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None, 2])),
    ).unbatch()

    validation_dataset = tf.data.Dataset.from_generator(
        lambda: make_generator(args.train_path + "test"),
        (tf.float32, tf.float32),
        (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None, 2])),
    ).unbatch()

    test_dataset = tf.data.Dataset.from_generator(
        lambda: make_generator(args.test_path),
        (tf.float32, tf.float32),
        (tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None, 2])),
    ).unbatch()

    config.train_sample_count = len(
        glob.glob(args.train_path + "train/**/*.*", recursive=True)
    )
    config.validation_sample_count = len(
        glob.glob(args.train_path + "test/**/*.*", recursive=True)
    )
    config.test_sample_count = len(glob.glob(args.test_path + "**/*.*", recursive=True))

    model = train(model, train_dataset, validation_dataset, config)
    test_loss, test_acc = test_eval(model, test_dataset, config)
    print("Test acc", test_acc)

    # print(test_pred(model, test_dataset, config))

    # model = train(model, train_dataset, validation_dataset, config)
    # test_loss, test_acc = test_eval(model, test_dataset, config)
    # print("Test acc", test_acc)

    # TODO: Remove subsetting after test results are satisfactory
    # test_confidence(model, validation_dataset.take(20), config)
    # print(test_pred(model, validation_dataset, config))


if __name__ == "__main__":
    if tf.config.list_physical_devices("GPU"):
        print("Using GPU")
    else:
        print("Using CPU")
    main()
