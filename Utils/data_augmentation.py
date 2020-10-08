import tensorflow as tf
import tensorflow_addons as tfa


""" 

Example    
- augmentented_images = images.map(lambda img: black_box_augmentation(img, 0.1))


TODO:
    - Add inverse functionality

"""

def black_box_augmentation(img, ratio):
    
    """ 
        Augment images with centered black box. 
        
            param: tensor img: A image in the form of a Tensorflow tensor
            param: float ratio: How large share of the image that should be augmented with a black box
    """
        
    X_len, Y_len, RGB = img.shape
    center_offset = ((X_len // 2), (Y_len // 2))
    ratio = tf.cast(ratio, tf.float32)
    box_area = tf.cast((tf.pow(X_len,2) ), tf.float32)
    square_len = tf.sqrt(box_area * ratio)
    square_len = tf.cast(square_len, tf.int32)
    
    img = tf.expand_dims(img, axis = 0)
    
    augmented_img = tfa.image.cutout(images = img, mask_size = square_len, offset = center_offset)
    
    return tf.squeeze(augmented_img)