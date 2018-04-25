# calls generator and discriminator
from network import *

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
import time
import random



outputsDiscFilter=64      # same  as no of discriminator filters
outputGenFilter=32        # same  as no of generator filters


def discriminator( imageDisc, name="discriminator"):
    with tf.variable_scope(name):
        #filter hieght and widht
        k_size=5
        imageDisc1=convolution(imageDisc, outputsDiscFilter, k_size, variable_name="convolution")
        imageDisc2=convolution(imageDisc1,outputsDiscFilter*2, k_size,variable_name="convolution")
        imageDisc3=convolution(imageDisc2,1,k_size,variable_name="convolution")

        return imageDisc3


def generator(imageGen, name="generator"):
    with tf.variable_scope(name):
        k_size=5

        #padding needed??
        #enconding
        imageGen1=convolution(imageGen,outputGenFilter,k_size,variable_name="generator")
        imageGen2=convolution(imageGen1,outputGenFilter*2,k_size-2,variable_name="generator")
        imageGen3=convolution(imageGen2, outputGenFilter*4, k_size-2,variable_name="generator")

        #transformation
        imageRes1=resnet(imageGen3,outputGenFilter*4,k_size-2,name="resnet")
        imageRes2=resnet(imageRes1,outputGenFilter*4, k_size-2,name="resnet")
        imageRes3 = resnet(imageRes2, outputGenFilter*4, k_size-2, name="resnet")
        imageRes4 = resnet(imageRes3,outputGenFilter*4, k_size-2, name="resnet")

        #decoding
        imageDecon1=deconvolution(imageRes4,outputGenFilter*2,k_size-2, name="deconvolution")
        imageDecon2=deconvolution(imageDecon1,outputGenFilter,k_size, name="deconvolution")

        # convolulation layer needed??
        # tanh needed??

        return imageDecon2


def generator_convolution(image, channels):

    gen_conv1_layer = convolution(image, outputGenFilter, 5)
    gen_conv2_layer = convolution(gen_conv1_layer, outputGenFilter, 3)
    gen_conv3_layer = convolution(gen_conv2_layer, outputGenFilter*2, 3)
    gen_conv4_layer = convolution(gen_conv3_layer, outputGenFilter*2, 3)

    gen_deconv1_layer = deconvolution(gen_conv4_layer, outputGenFilter*2, 3)
    gen_deconv2_layer = deconvolution(gen_deconv1_layer, outputGenFilter, 3)
    gen_image_layer = convolution(gen_deconv2_layer, channels, 3, tf.nn.sigmoid)

    return gen_image_layer

def discriminator_convolution(image,):
    disc_conv1_layer = convolution(image, outputsDiscFilter, 5)
    disc_conv2_layer = convolution(disc_conv1_layer, outputsDiscFilter, 3)
    disc_conv3_layer = convolution(disc_conv2_layer, outputsDiscFilter*2, 3)
    disc_conv4_layer = convolution(disc_conv3_layer, outputsDiscFilter*2, 3)
    output = convolution(disc_conv4_layer, 1, 3)

    return output
