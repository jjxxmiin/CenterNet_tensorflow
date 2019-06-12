# image augmentation

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def random_filp_lr(img):
    filped_img = tf.image.random_flip_left_right(img,seed=None)
    return filped_img


def random_filp_ud(img):
    filped_img = tf.image.random_flip_up_down(img,seed=None)
    return filped_img

def random_scaling(img):
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))

    new_shape = tf.squeeze(tf.stack([h_new,w_new]),squeeze_dims=[1])
    scaled_img = tf.image.resize_images(img,new_shape)

    return scaled_img