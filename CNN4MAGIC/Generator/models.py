import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import *
from keras.models import load_model, Model

# from CNN4MAGIC.CNN_Models.BigData.cbam_DenseNet import *
from CNN4MAGIC.CNN_Models.BigData.se_DenseNet import SEDenseNet, SEDenseNetImageNet121
from CNN4MAGIC.CNN_Models.BigData.se_resinc import SEInceptionResNetV2
from CNN4MAGIC.Generator.SqueezeExciteInceptionV3gencopy import SEInceptionV3
from CNN4MAGIC.Other_utilities.keras_efficientnets.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications import NASNetLarge, ResNet50


# %%

def squeeze_excite_block(input, ratio=10):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


# %%

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
def energy_skrr(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    out = Conv2D(250, (5, 5))(input_img)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = AveragePooling2D((3, 3))(out)

    out = Conv2D(150, (4, 4))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(100, (1, 1))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = AveragePooling2D((2, 2))(out)

    out = Conv2D(100, (4, 4))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(80, (1, 1))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(60, (1, 1))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = AveragePooling2D((2, 2))(out)

    out = Conv2D(40, (3, 3))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)

    out = Flatten()(out)

    # out = GlobalAveragePooling2D()(out)
    out = Dense(1, kernel_regularizer='l2')(out)

    model = Model(input_img, out)

    return model


# model = energy_skrr(True)
# model.compile('sgd', 'mse')
# model.summary()
# %%

def energy_skrr_se(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    out = Conv2D(250, (5, 5))(input_img)
    out = BatchNormalization()(out)
    # out = squeeze_excite_block(out)
    out = ELU()(out)
    out = AveragePooling2D((3, 3))(out)
    out = squeeze_excite_block(out)

    out = Conv2D(150, (4, 4))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(100, (1, 1))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(150, (4, 4))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(100, (1, 1))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = AveragePooling2D((2, 2))(out)
    out = squeeze_excite_block(out)

    out = Conv2D(100, (4, 4))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(80, (1, 1))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = squeeze_excite_block(out)

    out = Conv2D(100, (4, 4))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(80, (1, 1))(out)
    # out = squeeze_excite_block(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Conv2D(60, (1, 1))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = squeeze_excite_block(out)

    # out = AveragePooling2D((2, 2))(out)
    out = Flatten()(out)

    out = Conv2D(40, (3, 3))(out)
    out = BatchNormalization()(out)
    out = ELU()(out)

    out = Flatten()(out)

    # out = GlobalAveragePooling2D()(out)
    out = Dense(1, kernel_regularizer='l2')(out)

    model = Model(input_img, out)

    return model


# model = energy_skrr_se(True)
# model.compile('sgd', 'mse')
# model.summary()

# %%
def easy_dense(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    out = Conv2D(150, (5, 5))(input_img)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(150, (3, 3))(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    out = Conv2D(150, (1, 1))(out)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)

    dense_out = SEDenseNet(input_tensor=out,
                           include_top=False,
                           # depth=25,
                           nb_dense_block=3,
                           bottleneck=True,
                           growth_rate=12,
                           nb_filter=-1,
                           nb_layers_per_block=6,
                           dropout_rate=0,
                           weight_decay=1e-4)

    x = dense_out.layers[-1].output
    out = Dense(30)(x)
    out = BatchNormalization()(out)
    out = LeakyReLU()(out)
    final_out = Dense(1, kernel_regularizer='l2')(out)

    model = Model(input_img, final_out)

    return model


# model = easy_dense(True)
# model.compile('sgd', 'mse')
# model.summary()
# %%

def SE_InceptionV3_DoubleDense_energy(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    dense_out = SEInceptionV3(include_top=False,
                              weights=None,
                              input_tensor=input_img,
                              input_shape=None,
                              pooling='avg'
                              )
    x = dense_out.layers[-1].output

    x = BatchNormalization()(x)
    x = Dense(64, kernel_regularizer='l1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(32, kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


def SE_InceptionV3_SingleDense_energy(include_time=True, input=None):
    if input == None:
        if include_time:
            input_img = Input(shape=(67, 68, 4), name='m1m2')
        else:
            input_img = Input(shape=(67, 68, 2), name='m1m2')
    else:
        input_img = input

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
    # x = Dense(32, kernel_regularizer='l2')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


def SE_InceptionV3_SingleDense_direction(include_time=True, input=None):
    if input == None:
        if include_time:
            input_img = Input(shape=(67, 68, 4), name='m1m2')
        else:
            input_img = Input(shape=(67, 68, 2), name='m1m2')
    else:
        input_img = input

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
    # x = Dense(32, kernel_regularizer='l2')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Dense(2, name='direction', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


# %%

def SE_incres_TripleDense_energy(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    dense_out = SEInceptionResNetV2(include_top=False,
                                    weights=None,
                                    input_tensor=input_img,
                                    input_shape=None,
                                    pooling='avg'
                                    )
    x = dense_out.layers[-1].output

    x = BatchNormalization()(x)
    x = Dense(312)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # x = Dense(32, kernel_regularizer='l2')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


def SE_incres_energy(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    dense_out = SEInceptionResNetV2(include_top=False,
                                    weights=None,
                                    input_tensor=input_img,
                                    input_shape=None,
                                    pooling='avg'
                                    )
    x = dense_out.layers[-1].output

    x = BatchNormalization()(x)
    # x = Dense(312)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = Dense(128)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = Dense(32)(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = Dense(32, kernel_regularizer='l2')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    x = Dense(1, name='energy')(x)
    model1 = Model(inputs=input_img, output=x)

    return model1


# m = SE_incres_SingleDense_energy()
# m.summary()

# %%
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


def MobileNetV2_energy(alpha=1):
    input_img = Input(shape=(67, 68, 4), name='m1')

    model = MobileNetV2(alpha=alpha, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
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


def SEDenseNet121_position(input=None, include_time=True):
    if input is None:
        if include_time:
            input_img = Input(shape=(67, 68, 4), name='m1')
        else:
            input_img = Input(shape=(67, 68, 2), name='m1')
    else:
        input_img = input

    model = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None)

    x = model.layers[-1].output
    x = Dense(2, name='position')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def SEDenseNet121_position_l2(input=None, include_time=True):
    if input is None:
        if include_time:
            input_img = Input(shape=(67, 68, 4), name='m1')
        else:
            input_img = Input(shape=(67, 68, 2), name='m1')
    else:
        input_img = input
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


# %%
def single_DenseNet_25_3_doubleDense():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    # m1 = Input(shape=(67, 68, 2), name='m1')
    # m2 = Input(shape=(67, 68, 2), name='m2')
    # input_img = concatenate([m1, m2])
    dense_out = SEDenseNet(input_tensor=input_img,
                           include_top=False,
                           depth=25,
                           nb_dense_block=3,
                           dropout_rate=0)

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


# %%

def single_DenseNet_piccina():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=10, nb_dense_block=4, dropout_rate=0)

    x = dense_out.layers[-1].output
    x = Dense(1, name='energy', kernel_regularizer='l2')(x)

    model1 = Model(inputs=input_img, output=x)
    return model1


def small_SeDenseNet_separation():
    input_img = Input(shape=(67, 68, 4), name='m1m2')

    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=10, nb_dense_block=4, dropout_rate=0.4)

    x = dense_out.layers[-1].output
    x = Dense(1, name='gammaness')(x)

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


def MobileNetV2_4dense_energy_desperacion(input=None):
    if input is None:
        input_img = Input(shape=(67, 68, 4), name='m1')
    else:
        input_img = input

    model = MobileNetV2(alpha=3, depth_multiplier=1, include_top=False,
                        weights=None, input_tensor=input_img, pooling='avg')

    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(256, kernel_regularizer='l1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128, kernel_regularizer='l1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(64, kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(32, kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(1, name='energy', kernel_regularizer='l2')(x)
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


def MobileNetV2_separation(alpha=1.0, include_time=True, drop_rate=0.5):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = keras.applications.mobilenet_v2.MobileNetV2(alpha=alpha, include_top=False,
                                                        weights=None, input_tensor=input_img, pooling='max')

    x = model.layers[-1].output
    x = Dropout(drop_rate)(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B0_separation(include_time=True, drop_connect=0, dropout=0.2):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB0(input_tensor=input_img,
                           dropout_rate=dropout,
                           drop_connect_rate=drop_connect,
                           pooling='max',
                           include_top=False,
                           weights=None)
    x = model.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(3)(x)
    # x = Activation('selu')(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B1_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB1(input_tensor=input_img, dropout_rate=0.6, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B2_separation(include_time=True, dropout=0, drop_connect=0, last_is_three=False, nonlinear_last=False):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB2(input_tensor=input_img,
                           dropout_rate=dropout,
                           drop_connect_rate=drop_connect,
                           pooling='max',
                           include_top=False,
                           weights=None)
    x = model.layers[-1].output
    if last_is_three:
        x = BatchNormalization()(x)
        x = Dense(3)(x)
        if nonlinear_last:
            x = Activation('selu')(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B3_separation(include_time=True, dropout=0, drop_connect=0, last_is_three=False, nonlinear_last=False):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB3(input_tensor=input_img,
                           dropout_rate=dropout,
                           drop_connect_rate=drop_connect,
                           pooling='max',
                           include_top=False,
                           weights=None)
    x = model.layers[-1].output
    if last_is_three:
        x = BatchNormalization()(x)
        x = Dense(3)(x)
        if nonlinear_last:
            x = Activation('selu')(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B4_separation(include_time=True, dropout=0, drop_connect=0, last_is_three=False, nonlinear_last=False):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB4(input_tensor=input_img,
                           dropout_rate=dropout,
                           drop_connect_rate=drop_connect,
                           pooling='max',
                           include_top=False,
                           weights=None)
    x = model.layers[-1].output
    if last_is_three:
        x = BatchNormalization()(x)
        x = Dense(3)(x)
        if nonlinear_last:
            x = Activation('selu')(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def efficientNet_B5_separation(include_time=True, dropout=0, drop_connect=0, last_is_three=False, nonlinear_last=False):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = EfficientNetB5(input_tensor=input_img,
                           dropout_rate=dropout,
                           drop_connect_rate=drop_connect,
                           pooling='max',
                           include_top=False,
                           weights=None)
    x = model.layers[-1].output
    if last_is_three:
        x = BatchNormalization()(x)
        x = Dense(3)(x)
        if nonlinear_last:
            x = Activation('selu')(x)
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


# %%
def InceptionV3_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = InceptionV3(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


# %%
from keras.applications import ResNet50V2, ResNet101V2, Xception, DenseNet121, InceptionResNetV2, NASNetMobile, VGG16, \
    VGG19


def ResNet50V2_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = ResNet50V2(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def VGG16_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = VGG16(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def VGG19_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = VGG19(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def ResNet101V2_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = ResNet101V2(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def Xception_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = Xception(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def DenseNet121_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = DenseNet121(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def InceptionResNetV2_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = InceptionResNetV2(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def NASNetMobile_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = NASNetMobile(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def NASNet_separation(include_time=True):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    model = NASNetLarge(input_tensor=input_img, pooling='max', include_top=False, weights=None)
    x = model.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


from CNN4MAGIC.Other_utilities.coord import CoordinateChannel2D


def coord_conv2d(input, filters, kernel_size, act):
    x = CoordinateChannel2D()(input)
    x = Conv2D(filters, kernel_size, activation=act)(x)
    return x


def vgg_like_position_net(include_time=True, depth_multiplier=3, do_batchnorm=False):
    if include_time:
        input_img = Input(shape=(67, 68, 4), name='m1m2')
    else:
        input_img = Input(shape=(67, 68, 2), name='m1m2')

    x = coord_conv2d(input_img, 64, (3, 3), 'selu')
    x = coord_conv2d(x, 64, (3, 3), 'selu')
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    if do_batchnorm:
        x = BatchNormalization()(x)

    # Block 2
    x = coord_conv2d(x, 128, (3, 3), 'selu')
    x = coord_conv2d(x, 128, (3, 3), 'selu')
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    for _ in range(depth_multiplier):
        x = coord_conv2d(x, 256, (3, 3), 'selu')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    for _ in range(depth_multiplier):
        x = coord_conv2d(x, 512, (3, 3), 'selu')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    for _ in range(depth_multiplier):
        x = coord_conv2d(x, 512, (3, 3), 'selu')
    x = GlobalMaxPool2D()(x)

    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=input_img, output=x)
    return model1


def pos_vgg_like_16_separation():
    return vgg_like_position_net(include_time=True, depth_multiplier=3, do_batchnorm=False)


def pos_vgg_like_19_separation():
    return vgg_like_position_net(include_time=True, depth_multiplier=4, do_batchnorm=False)


def pos_vgg_like_16_bn_separation():
    return vgg_like_position_net(include_time=True, depth_multiplier=3, do_batchnorm=True)


def pos_vgg_like_19_bn_separation():
    vgg_like_position_net(include_time=True, depth_multiplier=4, do_batchnorm=True)
    return


def pos_vgg_like_21_separation():
    return vgg_like_position_net(include_time=True, depth_multiplier=5, do_batchnorm=False)


def pos_vgg_like_24_separation():
    return vgg_like_position_net(include_time=True, depth_multiplier=6, do_batchnorm=False)


from keras import layers


def modded_vgg19(drop_rate=0., pooling='max'):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    input_img = Input(shape=(67, 68, 4), name='m1m2')
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(input_img)
    x = Dropout(drop_rate)(x)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Dropout(drop_rate)(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Dropout(drop_rate)(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)

    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)

    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)

    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)

    x = Dropout(drop_rate)(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    x = Dense(1, activation='sigmoid', name='gammaness')(x)

    model = Model(input_img, x, name='vgg19_mine')

    return model
