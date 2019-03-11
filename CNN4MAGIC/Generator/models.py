import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.cbam_DenseNet import *
from CNN4MAGIC.CNN_Models.BigData.se_DenseNet import SEDenseNet, SEDenseNetImageNet121
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


def NASNet_mobile_position(input=None):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = keras.applications.nasnet.NASNetMobile(include_top=False, weights=None, input_tensor=input_img,
                                                   pooling='avg')

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def NASNet_mobile_energy():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = keras.applications.nasnet.NASNetMobile(include_top=False, weights=None, input_tensor=input_img,
                                                   pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
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


def SEDenseNet121_position(input=None):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None)

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def SEDenseNet121_position_l2():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None)

    x = model.layers[-1].output
    x = Dense(2, name='position', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def SEDenseNet121_position_l2_drop02(drop=0.2):
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None, dropout_rate=drop)

    x = model.layers[-1].output
    x = Dense(2, name='position', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def SEDenseNet121_energy():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None)

    x = model.layers[-1].output
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def SEDenseNet121_energy_dropout_l2(drop=0.2):
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None, dropout_rate=drop)

    x = model.layers[-1].output
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def InceptionV3_position():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = InceptionV3(include_top=False, weights=None, input_tensor=input_img,
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


def single_DenseNet_piccina():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=10, nb_dense_block=4, dropout_rate=0)

    x = dense_out.layers[-1].output
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)

    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_4dense_position(input=None):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def Slim_MobileNetV2_4dense_position(input=None):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def dummy_cnn():
    # input_img = Input(shape=(67, 68, 2), name='m1m2')

    input_img = Input(shape=(67, 68, 2), name='m1m2')

    x = Conv2D(4, (20, 20), strides=(2, 2), use_bias=False)(input_img)
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    # x = Conv2D(50, (1, 1))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    # x = MaxPooling2D((3, 3))(x)
    # x = Conv2D(40, (3, 3))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    # x = MaxPooling2D((3,3))(x)
    # x = Conv2D(40, (3, 3))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = Conv2D(40, (3, 3))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    # x = MaxPooling2D((3,3))(x)
    # x = Conv2D(40, (3,3))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    x = GlobalMaxPool2D()(x)
    # x = Dense(20)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)

    out = Dense(1, activation='sigmoid', use_bias=False)(x)

    model_dummy = Model(input_img, out)

    return model_dummy


def dummy_cnn_2filter5():
    input_img = Input(shape=(67, 68, 2), name='m1m2')

    x = Conv2D(2, (5, 5), strides=(2, 2), use_bias=False)(input_img)
    x = ReLU()(x)
    x = GlobalMaxPool2D()(x)

    out = Dense(1, activation='sigmoid', use_bias=False)(x)

    model_dummy = Model(input_img, out)

    return model_dummy


def MobileNetV2_separation(alpha=1.0, include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = keras.applications.mobilenet_v2.MobileNetV2(alpha=alpha, depth_multiplier=1, include_top=False,
                                                        weights=None, input_tensor=input_img, pooling='max')

    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_4dense_energy_dropout():
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output

    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_2dense_energy(pretrained=False, drop=False, freeze_cnn=False):
    input_img = Input(shape=(67, 68, 4), name='m1')

    if pretrained:
        path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5'
        model = load_model(path)
        input_img = model.layers[0].input
        x = model.layers[-15].output  # This is the output of the global avg pooling
        if freeze_cnn:
            for layer in model.layers[:-15]:
                layer.trainable = False
    else:
        model = MobileNetV2(alpha=0.3, depth_multiplier=0.3, include_top=False,
                            weights=None, input_tensor=input_img, pooling='avg')


        x = model.layers[-1].output

    x = BatchNormalization()(x)
    if drop:
        x = Dropout(.4)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    if drop:
        x = Dropout(.4)(x)
    x = LeakyReLU()(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    if drop:
        x = Dropout(.4)(x)
    x = LeakyReLU()(x)

    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def MobileNetV2_4dense_energy(pretrained=False, drop=False, freeze_cnn=False, input=None):
    if input is not None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    if pretrained:
        path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5'
        model = load_model(path)
        input_img = model.layers[0].input
        x = model.layers[-15].output  # This is the output of the global avg pooling
        if freeze_cnn:
            for layer in model.layers[:-15]:
                layer.trainable = False
    else:
        model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
                            weights=None, input_tensor=input_img, pooling='avg')

        x = model.layers[-1].output

    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


# def MobileNetV2_4dense_energy():
#     input_img = Input(shape=(67, 68, 4), name='m1')
#
#     model = MobileNetV2(alpha=1, depth_multiplier=1, include_top=False,
#                         weights=None, input_tensor=input_img, pooling='avg')
#
#     x = model.layers[-1].output
#
#     x = BatchNormalization()(x)
#     x = Dense(256)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(64)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(32)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(1, name='energy')(x)
#     model1 = Model(inputs=input_img, output=x)
#     return model1




# def MobileNetV2_2dense_energy(pretrained=False, drop=False):
#     input_img = Input(shape=(67, 68, 4), name='m1')
#
#     if pretrained:
#         path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5'
#         model = load_model(path)
#         input_img = model.layers[0].input
#         x = model.layers[-15].output  # This is the output of the global avg pooling
#     else:
#         model = NASNet_mobile_position(include_top=False,
#                                        weights=None, input_tensor=input_img, pooling='avg')
#
#         x = model.layers[-1].output
#
#     x = BatchNormalization()(x)
#     if drop:
#         x = Dropout(.4)(x)
#     x = Dense(128)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(.4)(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(64)(x)
#     x = BatchNormalization()(x)
#     if drop:
#         x = Dropout(.4)(x)
#     x = LeakyReLU()(x)
#
#     x = Dense(1, name='energy')(x)
#     model1 = Model(inputs=input_img, output=x)
#     return model1


def Slim_MobileNetV2_2dense_position(input=None, alpha=0.1, depth_m=1):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = MobileNetV2(alpha=alpha, depth_multiplier=depth_m, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1
