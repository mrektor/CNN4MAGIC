import keras
from keras.applications.mobilenetv2 import MobileNetV2

from CNN4MAGIC.CNN_Models.BigData.cbam_DenseNet import *
from CNN4MAGIC.CNN_Models.BigData.se_DenseNet import SEDenseNet
from CNN4MAGIC.Generator.SqueezeExciteInceptionV3gencopy import SEInceptionV3


def MobileNetV2_slim():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=0.5, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='max')

    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


# %%
# model = MobileNetV2_slim()
# #%%
# model.compile('sgd', 'mse')
# #%%
# import numpy as np
# x= np.random.randn(1000, 67, 68, 4)
# y= np.random.randn(1000, 1)
#
# #%%
# model.fit(batch_size=64, epochs=3, x=x, y=y)

# %%


# %%

def SE_InceptionV3_DoubleDense_energy():
    input_img = Input(shape=(67, 68, 4), name='m1')

    dense_out = SEInceptionV3(include_top=False,
                              weights=None,
                              input_tensor=input_img,
                              input_shape=None,
                              pooling='avg'
                              )
    x = dense_out.layers[-1].output

    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


def MobileNetV2_slim_energy():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=0.5, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='max')

    x = model.layers[-1].output
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_slim_position():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=0.5, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_position():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_energy():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1

def MobileNetV2_energy_doubleDense():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1



def NASNet_mobile_position():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = keras.applications.nasnet.NASNetMobile(include_top=False, weights=None, input_tensor=input_img,
                                                   pooling='avg')

    x = model.layers[-1].output
    x = Dense(2, name='position', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def DenseNet121_position():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=input_img,
                                                    pooling='avg')

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def CBAM_DenseNet161_Energy():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    model = CBAMDenseNetImageNet161(input_tensor=input_img, include_top=False, weights=None)

    x = model.layers[-1].output
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def CBAM_DenseNet121_Energy():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    model = CBAMDenseNetImageNet121(input_tensor=input_img, include_top=False, dropout_rate=0.35, weights=None)

    x = model.layers[-1].output
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def single_DenseNet_25_3_doubleDense():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    # m1 = Input(shape=(67, 68, 2), name='m1')
    # m2 = Input(shape=(67, 68, 2), name='m2')
    # input_img = concatenate([m1, m2])
    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=25, nb_dense_block=3, dropout_rate=0)

    x = dense_out.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1
