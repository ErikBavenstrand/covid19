import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import math
import ntpath
import platform

import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from Utils import make_dir

# Default values
AUTO = tf.data.experimental.AUTOTUNE
WIDTH = 224
HEIGHT = 224
SYSTEM = platform.system()

# Initial dataset https://data.mendeley.com/datasets/8h65ywd2jr/3


######################################################
# Helper functions for reading and writing TFRecords #
######################################################
def _float32_list(floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


def _int64_list(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _byte_list(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfrecord(tfrec_filewriter, image, label, file_name):
    feature = {
        'image': _byte_list(image),
        'label': _int64_list(label),
        'file_name': _byte_list(file_name),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _encode_image_tfrecord_linux(file_name):
    class_folders = tf.constant([
        '\.\/COVID-19 Dataset\/X-ray\/Non-COVID.+',
        '\.\/COVID-19 Dataset\/X-ray\/COVID.+'
    ])

    return _encode_image_tfrecord_common(class_folders, file_name)


def _encode_image_tfrecord_windows(file_name):
    class_folders = tf.constant([
        '\.\\\\COVID-19 Dataset\\\\X-ray\\\\Non-COVID.+',
        '\.\\\\COVID-19 Dataset\\\\X-ray\\\\COVID.+'
    ])

    return _encode_image_tfrecord_common(class_folders, file_name)


def _encode_image_tfrecord_common(class_folders, file_name):
    image_label = tf.cast(
        tf.map_fn(lambda x: tf.strings.regex_full_match(file_name, x),
                  class_folders,
                  fn_output_signature=tf.bool), tf.int64)
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])
    image = tf.cast(image, np.uint8)
    image = tf.image.encode_jpeg(image)
    return image, image_label, file_name


def _decode_image_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([2], tf.int64),
        'file_name': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3) / 255
    label = example['label']
    file_name = example['file_name']

    return image, label


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def rename_files_enumerate(base_path):
    all_files = [
        y for x in os.walk(base_path) for y in glob(os.path.join(x[0], '*.*'))
    ]
    for i, file in enumerate(
            tqdm(all_files, desc='Enumerating files and renaming')):
        filename = path_leaf(file)
        filename = filename.split('.' + filename.split('.')[-1])[0]
        new_name = file.replace(filename, str(i) + '-mendeley')
        os.rename(file, new_name)


def convert_images_to_jpg(base_path):
    jpg_files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.jpg'))
    ]
    if jpg_files:
        for file in tqdm(jpg_files, desc='Renaming .jpg to .jpeg'):
            os.rename(file, file.replace('.jpg', '.jpeg'))

    JPG_files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.JPG'))
    ]
    if JPG_files:
        for file in tqdm(JPG_files, desc='Renaming .JPG to .jpeg'):
            os.rename(file, file.replace('.JPG', '.jpeg'))

    jfif_files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.jfif'))
    ]
    if jfif_files:
        for file in tqdm(jfif_files, desc='Renaming .jfif to .jpeg'):
            os.rename(file, file.replace('.jfif', '.jpeg'))

    png_files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.png'))
    ]
    if png_files:
        for file in tqdm(png_files, desc='Converting .png to .jpeg'):
            im = cv2.imread(file)
            cv2.imwrite(file.replace('.png', '.jpeg'), im)
            os.remove(file)

    PNG_files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.PNG'))
    ]
    if PNG_files:
        for file in tqdm(PNG_files, desc='Converting .PNG to .jpeg'):
            im = cv2.imread(file)
            cv2.imwrite(file.replace('.PNG', '.jpeg'), im)
            os.remove(file)

    files = [
        y for x in os.walk(base_path)
        for y in glob(os.path.join(x[0], '*.jpeg'))
    ]
    if files:
        for file in tqdm(files, desc='Converting images to b&w'):
            im = cv2.imread(file)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(file, im)


def generate_tfrecord_files(tfrecords_path, images_path, images_per_file):
    make_dir(tfrecords_path)

    images_path_pattern = images_path + '*/*.jpeg'
    found_images = len(tf.io.gfile.glob(images_path_pattern))
    print(
        'Pattern matches {} images which will be rewritten as {} TFRecord files containing ~{} images each.'
        .format(found_images, math.ceil(found_images / images_per_file),
                images_per_file))
    images = tf.data.Dataset.list_files(images_path_pattern)
    if SYSTEM == 'Linux' or SYSTEM == 'Darwin':
        encode_image = _encode_image_tfrecord_linux
    elif SYSTEM == 'Windows':
        encode_image = _encode_image_tfrecord_windows

    dataset = images.map(encode_image,
                         num_parallel_calls=AUTO).batch(images_per_file)

    for file_number, (image, label, file_name) in enumerate(
            tqdm(dataset, desc='Generating TFRecords')):
        tfrecord_filename = tfrecords_path + "{:02d}-{}.tfrecord".format(
            file_number, images_per_file)

        images_in_this_file = image.numpy().shape[0]
        if not os.path.isfile(tfrecord_filename):
            with tf.io.TFRecordWriter(tfrecord_filename) as out_file:
                for i in range(images_in_this_file):
                    example = _to_tfrecord(out_file,
                                           np.array(image)[i],
                                           label.numpy()[i],
                                           file_name.numpy()[i])
                    out_file.write(example.SerializeToString())


def read_tfrecord_files(tfrecords_path='./dataset/train/'):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    filenames_tf = tf.io.gfile.glob(tfrecords_path + "*.tfrecord")
    dataset = tf.data.TFRecordDataset(filenames_tf, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(_decode_image_tfrecord, num_parallel_calls=AUTO)
    return dataset


def get_tfrecord_sample_count(tfrecords_path='./dataset/train/'):
    filenames_tf = tf.io.gfile.glob(tfrecords_path + "*.tfrecord")
    count = sum(1 for _ in tf.data.TFRecordDataset(filenames_tf,
                                                   num_parallel_reads=AUTO))
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TFRecords from images')
    parser.add_argument(
        '--images-path',
        type=str,
        default='./COVID-19 Dataset/X-ray/',
        metavar='PATH',
        help=
        'images path to generate records from (default: ./COVID-19 Dataset/X-ray/)'
    )
    parser.add_argument('--tfrecords-path',
                        type=str,
                        default='./dataset/train/',
                        metavar='PATH',
                        help='tfrecord path (default: ./dataset/train/)')
    parser.add_argument('--images-per-file',
                        type=int,
                        default=2048,
                        metavar='N',
                        help='images per tfrecord file (default: 2048)')
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
    parser.add_argument('--skip-conversion',
                        default=False,
                        action='store_true',
                        help='skip conversion and renaming of images')
    args = parser.parse_args()

    WIDTH = args.image_width
    HEIGHT = args.image_height

    if not args.skip_conversion:
        rename_files_enumerate(args.images_path)
        convert_images_to_jpg(args.images_path)

    generate_tfrecord_files(args.tfrecords_path, args.images_path,
                            args.images_per_file)
