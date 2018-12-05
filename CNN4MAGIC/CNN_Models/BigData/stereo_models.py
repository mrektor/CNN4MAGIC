import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Add, DepthwiseConv2D
from keras.layers import Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU
from keras.models import Model


def feats(x_train, input_shape, baseDim=16, padding="valid", dropout=0.1):
    out = Conv2D(baseDim * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding,
                 input_shape=input_shape)(x_train)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    # out = MaxPooling2D(pool_size=(2, 2)(out))

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    # out = MaxPooling2D(pool_size=(2, 2)(out))

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 8 * 2 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)

    out = GlobalAveragePooling2D()(out)

    return out


def energy_stereo_time_v1():
    m1 = Input(shape=(67, 68, 2), name='m1')
    m2 = Input(shape=(67, 68, 2), name='m2')

    out_a = feats(m1, (67, 68, 2))
    out_b = feats(m2, (67, 68, 2))

    concatenated = keras.layers.concatenate([out_a, out_b])

    out = Dense(60, activation='relu')(concatenated)
    out = Dropout(0.2)(out)
    out = Dense(20, activation='relu')(out)
    out = Dense(1)(out)

    energy_regressor = Model(inputs=[m1, m2], outputs=out)

    return energy_regressor


def bottleneck_block(x, expand=64, squeeze=16, stride=(1, 1)):
    m = Conv2D(expand, (1, 1))(x)
    m = BatchNormalization()(m)
    m = ReLU()(m)
    m = DepthwiseConv2D((3, 3), strides=stride)(m)
    m = BatchNormalization()(m)
    m = ReLU()(m)
    m = Conv2D(squeeze, (1, 1))(m)
    m = BatchNormalization()(m)
    return Add()([m, x])


def stem_mobilenetV2(input):
    out = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(input)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = MaxPooling2D((3, 3))(out)

    out = bottleneck_block(out)

    for _ in range(2):
        out = bottleneck_block(out, expand=24 * 5, squeeze=24, stride=(2, 2))

    for _ in range(3):
        out = bottleneck_block(out, expand=32 * 5, squeeze=32, stride=(2, 2))

    for _ in range(4):
        out = bottleneck_block(out, expand=64 * 5, squeeze=64, stride=(2, 2))

    out = GlobalAveragePooling2D()(out)

    return out


def simple_mono(input):
    out = stem_mobilenetV2(input)
    out = Dense(32)(out)
    out = Dense(1)(out)
    return out
