import os
import errno

import numpy as np


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def blacken_img_section(img, x_section, y_section, x_sections, y_sections):
    """
        Augment images with black box in grid.

            param:

    """
    IMG_SHAPE = (224, 224, 3)
    img.set_shape(IMG_SHAPE)
    x_len, y_len, RGB = img.get_shape()

    x_pxs = x_len // x_sections
    y_pxs = y_len // y_sections

    x_start = x_pxs * x_section
    x_end = x_pxs * (x_section + 1)
    y_start = y_pxs * y_section
    y_end = y_pxs * (y_section + 1)

    mask = np.ones(IMG_SHAPE)
    mask[x_start:x_end, y_start:y_end, :] = 0
    new_img = img * mask
    return new_img
