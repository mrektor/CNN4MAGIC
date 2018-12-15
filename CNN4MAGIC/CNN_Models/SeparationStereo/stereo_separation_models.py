from keras.layers import *
from keras.models import Model

from CNN4MAGIC.CNN_Models.BigData.se_DenseNet import SEDenseNet


def single_DenseNet_25_3_doubleDense():
    m1 = Input(shape=(67, 68, 2), name='m1')
    m2 = Input(shape=(67, 68, 2), name='m2')
    input_img = concatenate([m1, m2])
    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=25, nb_dense_block=3, dropout_rate=0)

    x = dense_out.layers[-1].output
    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(1, name='gammaness', activation='sigmoid')(x)
    model1 = Model(inputs=[m1, m2], output=x)
    return model1


def single_DenseNet_piccina():
    m1 = Input(shape=(67, 68, 2), name='m1')
    m2 = Input(shape=(67, 68, 2), name='m2')
    input_img = concatenate([m1, m2])
    dense_out = SEDenseNet(input_tensor=input_img, include_top=False, depth=10, nb_dense_block=4, dropout_rate=0)

    x = dense_out.layers[-1].output
    x = Dense(1, name='gammaness', activation='sigmoid')(x)

    model1 = Model(inputs=[m1, m2], output=x)
    return model1
