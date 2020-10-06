import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from Models import simple_cnn, vgg16
from Utils import grad_cam
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from Utils import make_dir

WIDTH = 224
HEIGHT = 224


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer
    raise ValueError("Could not find conv layer. Cannot apply GradCAM.")


def create_image_array(folder_path):

    files = os.listdir(folder_path)
    n_files = len(files)
    image_array = np.empty((n_files, 224, 224, 3))
    file_names = []

    for image_number, image_path in enumerate(files):
        rescaled_image = load_img(folder_path + image_path,
                                  target_size=(WIDTH, HEIGHT, 3))
        image_array[image_number] = img_to_array(rescaled_image) / 255
        file_names.append(image_path)

    return image_array, file_names


def make_gradcam_heatmap(images_array, model, classifier_layer_names):
    last_conv_layer = find_last_conv_layer(model)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Create model that maps the activations of the final conv layer
    # to the final predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(images_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds, axis=1)
        one_hot_mask = tf.one_hot(top_pred_index,
                                  preds.shape[1],
                                  on_value=True,
                                  off_value=False,
                                  dtype=tf.bool)
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

    #heatmap = np.where(heatmap < 0.3, 0, heatmap)
    return heatmap


def combine_heatmap_image(image_array, heatmap, file_names):
    image_array = np.uint8(255 * image_array)
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap('inferno')

    jet_colors = jet(np.arange(256))[:, :3]

    for hm, im, name in zip(heatmap, image_array, file_names):
        jet_heatmap = jet_colors[hm]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(
            (image_array.shape[1], image_array.shape[2]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap + im
        superimposed_img = keras.preprocessing.image.array_to_img(
            superimposed_img)

        folder = './Grad Cam Output'
        make_dir(folder)
        save_path = folder + '/' + name
        superimposed_img.save(save_path)


def generate_gradcam(folder_path, model, classifier_layer_names):
    image_array, file_names = create_image_array(folder_path)
    heatmap = make_gradcam_heatmap(image_array, model, classifier_layer_names)
    combine_heatmap_image(image_array, heatmap, file_names)


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument('--model-name',
                        type=str,
                        metavar='FILENAME',
                        help='filename of model weights')
    parser.add_argument('--image-width',
                        type=int,
                        default=224,
                        metavar='N',
                        help='width of image (default: 224)')
    parser.add_argument('--image-height',
                        type=int,
                        default=224,
                        metavar='N',
                        help='height of image (default: 224)')
    args = parser.parse_args()

    WIDTH = args.image_width
    HEIGHT = args.image_height
    MODEL_PATH = './Saved Models/'

    model = load_model(MODEL_PATH + args.model_name)

    generate_gradcam('./test/', model, [
        'flatten', 'dense', 'activation_3', 'dropout', 'dense_1',
        'activation_4'
    ])


if __name__ == "__main__":
    main()
