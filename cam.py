import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from Utils import make_dir
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Models import simple_cnn, vgg16, vgg16cam
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    save_img,
    array_to_img,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model, Input
from tensorflow.keras.backend import function
import tensorflow as tf
import numpy as np
import argparse
import math


WIDTH = 224
HEIGHT = 224


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, Conv2D):
            return layer
    raise ValueError("Could not find conv layer. Cannot apply GradCAM.")


def create_image_array(folder_path):

    files = os.listdir(folder_path)
    n_files = len(files)
    image_array = np.empty((n_files, 224, 224, 3))
    file_names = []

    for image_number, image_path in enumerate(files):
        rescaled_image = load_img(
            folder_path + image_path, target_size=(WIDTH, HEIGHT, 3)
        )
        image_array[image_number] = img_to_array(rescaled_image) / 255
        file_names.append(image_path)

    return image_array, file_names


# BATCH SIZE NOT IMPLEMENTED
def make_gradcam_heatmap(images_array, model, batch_size):
    last_conv_layer = find_last_conv_layer(model)

    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    after_last_conv = False
    classifier_layer_names = []
    for layer in model.layers:
        if after_last_conv:
            classifier_layer_names.append(layer.name)
        elif find_last_conv_layer(model).name == layer.name:
            after_last_conv = True

    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(images_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds, axis=1)
        one_hot_mask = tf.one_hot(
            top_pred_index,
            preds.shape[1],
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )
        top_class_channel = tf.boolean_mask(preds, one_hot_mask)

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[0]):
        for j in range(pooled_grads.shape[-1]):
            last_conv_layer_output[i, :, :, j] *= pooled_grads[i, j]

    heatmap = np.mean(last_conv_layer_output, axis=-1)

    max_heatmap = np.maximum(heatmap, 0)
    for i in range(pooled_grads.shape[0]):
        heatmap[i] = max_heatmap[i] / np.max(heatmap[i])

    # heatmap = np.where(heatmap < 0.3, 0, heatmap)
    return heatmap


def combine_heatmap_image(image_array, heatmap, file_names, folder_path):
    image_array = np.uint8(255 * image_array)
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("inferno")

    jet_colors = jet(np.arange(256))[:, :3]

    for hm, im, name in zip(heatmap, image_array, file_names):
        jet_heatmap = jet_colors[hm]

        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image_array.shape[1], image_array.shape[2]))
        jet_heatmap = img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.7 + im
        superimposed_img = array_to_img(superimposed_img)

        make_dir(folder_path)
        save_path = folder_path + name
        superimposed_img.save(save_path)


# Need to add support for both classes in some way
def make_cam_heatmap(images_array, model, batch_size):
    class_weights = model.layers[-1].get_weights()[0]
    last_conv_layer = find_last_conv_layer(model)

    get_output = function([model.input], [last_conv_layer.output, model.output])

    batches = np.array_split(
        images_array, math.ceil(images_array.shape[0] / batch_size)
    )

    heatmap = np.zeros(
        dtype=np.float32,
        shape=(images_array.shape[0], *last_conv_layer.output.shape[1:3]),
    )

    for i, batch in enumerate(batches):
        images_in_batch = batch.shape[0]
        current_index = i * images_in_batch
        [conv_outputs, predictions] = get_output(batch)

        target_class = 1
        for j, w in enumerate(class_weights[:, target_class]):
            heatmap[current_index : current_index + images_in_batch] += (
                w * conv_outputs[:, :, :, j]
            )

    max_heatmap = np.maximum(heatmap, 0)
    for i in range(images_array.shape[0]):
        heatmap[i] = max_heatmap[i] / np.max(heatmap[i])

    return heatmap


def generate_gradcam(args, model):
    image_array, file_names = create_image_array(args.test_path)
    heatmap = make_gradcam_heatmap(image_array, model, args.batch_size)
    combine_heatmap_image(image_array, heatmap, file_names, "./gradcam output/")


def generate_cam(args, model):
    image_array, file_names = create_image_array(args.test_path)
    heatmap = make_cam_heatmap(image_array, model, args.batch_size)
    combine_heatmap_image(image_array, heatmap, file_names, "./cam output/")


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument(
        "--model-name", type=str, metavar="FILENAME", help="filename of model weights"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        metavar="N",
        help="width of image (default: 224)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        metavar="N",
        help="height of image (default: 224)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="./dataset/test/",
        metavar="PATH",
        help="path to testset images (default: ./dataset/test/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--grad-cam", action="store_true", help="run GradCAM instead of CAM"
    )
    args = parser.parse_args()

    WIDTH = args.image_width
    HEIGHT = args.image_height
    MODEL_PATH = "./Saved Models/"

    model = load_model(MODEL_PATH + args.model_name)

    if args.grad_cam:
        generate_gradcam(args, model)
    else:
        generate_cam(args, model)


if __name__ == "__main__":
    main()
