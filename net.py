"""
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""

import os
# change TensorFlow logging verbosity
# 0: all logs
# 1: filter out INFO logs
# 2: filter out WARNINGS logs
# 3: filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# use the GPU with the lowest memory usage
import GPUtil
DEVICE_ID = str(GPUtil.getFirstAvailable(order='memory')[0])
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
print('GPU selected:', DEVICE_ID)


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras.backend as K
K.set_session(session)

from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, MaxPooling2D, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD


class Settings:
    def __init__(self):
        self.patience = 10
        self.epochs = 100
        self.lr_factor = 0.1
        self.batch_size = 128


def Silvia1(patch_size, opt=SGD()):
    """
    LeNet with:
        2 convolutional layers with 20 kernels of size 5x5, ReLU activation and 2x2 max-pooling
        2 fully connected layers of 500 and 2 neurons, respectively
    """
    inputs = Input(shape=patch_size, name='in_0')
    norm = Lambda(lambda x : (x-128)*0.0125)(inputs)
    c1 = Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation=None, padding='valid', name='conv_0')(norm)
    mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_0')(c1)
    c2 = Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation=None, padding='valid', name='conv_1')(mp1)
    mp2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_1')(c2)
    f1 = Flatten(name='flatten')(mp2)
    d1 = Dense(500, activation='relu', name='relu')(f1)
    out = Dense(2, activation='softmax', name='softmax')(d1)
    model = Model(inputs=inputs, outputs=out)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model



def Silvia2(patch_size, opt=SGD()):
    """
    LeNet with:
        2 convolutional layers with 20 kernels of size 3x3, ReLU activation and 2x2 max-pooling
        2 fully connected layers of 250 and 2 neurons, respectively
    """
    inputs = Input(shape=patch_size, name='in_0')
    norm = Lambda(lambda x: (x - 128) * 0.0125)(inputs)
    c1 = Conv2D(20, kernel_size=(3, 3), strides=(1, 1), activation=None, padding='valid', name='conv_0')(norm)
    mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_0')(c1)
    c2 = Conv2D(50, kernel_size=(3, 3), strides=(1, 1), activation=None, padding='valid', name='conv_1')(mp1)
    mp2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_1')(c2)
    f1 = Flatten(name='flatten')(mp2)
    d1 = Dense(250, activation='relu', name='relu')(f1)
    out = Dense(2, activation='softmax', name='softmax')(d1)
    model = Model(inputs=inputs, outputs=out)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model


def Silvia3(patch_size, opt=SGD()):
    """
    LeNet with:
        1 convolutional layer with 20 kernels of size 5x5, ReLU activation and 2x2 max-pooling
        2 fully connected layers of 500 and 2 neurons, respectively
    """
    inputs = Input(shape=patch_size, name='in_0')
    norm = Lambda(lambda x: (x - 128) * 0.0125)(inputs)
    c1 = Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation=None, padding='valid', name='conv_0')(norm)
    mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='max_0')(c1)
    f1 = Flatten(name='flatten')(mp1)
    d1 = Dense(500, activation='relu', name='relu')(f1)
    out = Dense(2, activation='softmax', name='softmax')(d1)
    model = Model(inputs=inputs, outputs=out)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model


def Auto1(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 32 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(8, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto2(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 16 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)
    x = Conv2D(16, (2, 2), strides=2, padding='same')(x)

    enc = Conv2D(16, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto3(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 64 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(16, (2, 2), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder







def Auto3D1(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 32 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(8, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder


def Auto3D2(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 16 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)
    x = Conv2D(16, (2, 2), strides=2, padding='same')(x)

    enc = Conv2D(16, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto3D3(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 64 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(16, (2, 2), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder
