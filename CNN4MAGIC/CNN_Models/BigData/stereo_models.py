import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Add, DepthwiseConv2D
from keras.layers import Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU, LeakyReLU, add, concatenate
from keras.models import Model


# %%

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


#%%
def bottleneck_block(x, expand=64, squeeze=16, stride=(1, 1)):
    m = Conv2D(expand, (1, 1))(x)
    m = BatchNormalization()(m)
    m = ReLU()(m)
    m = DepthwiseConv2D((3, 3), strides=stride, padding='same')(m)
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

    out1 = MaxPooling2D((3, 3))(out)

    out = bottleneck_block(out1, squeeze=32)

    for _ in range(5):
        out = bottleneck_block(out, expand=24 * 5, squeeze=32, stride=(1, 1))

    out = MaxPooling2D((2, 2))(out)

    for _ in range(5):
        out = bottleneck_block(out, expand=32 * 5, squeeze=32, stride=(1, 1))

    out = MaxPooling2D((2, 2))(out)

    for _ in range(5):
        out = bottleneck_block(out, expand=64 * 5, squeeze=32, stride=(1, 1))

    out = MaxPooling2D((2, 2))(out)

    for _ in range(2):
        out = bottleneck_block(out, expand=64 * 5, squeeze=32, stride=(1, 1))

    out = GlobalAveragePooling2D()(out)

    return out


def magic_mobile_doubleStem():
    m1 = Input(shape=(67, 68, 2), name='m1')
    m2 = Input(shape=(67, 68, 2), name='m2')

    last_out_1 = stem_mobilenetV2(m1)
    last_out_2 = stem_mobilenetV2(m2)
    concatenated = keras.layers.concatenate([last_out_1, last_out_2])
    out = Dense(64)(concatenated)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = Dense(10)(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    very_out = Dense(1)(out)

    energy_regressor = Model([m1, m2], outputs=very_out)
    return energy_regressor


def magic_mobile_singleStem():
    m1 = Input(shape=(67, 68, 2), name='m1')
    m2 = Input(shape=(67, 68, 2), name='m2')

    input_img = concatenate([m1, m2])

    last_out = stem_mobilenetV2(input_img)

    out = Dense(64)(last_out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = Dense(10)(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    very_out = Dense(1)(out)

    energy_regressor = Model([m1, m2], outputs=very_out)
    return energy_regressor


def common_shit(input_layer, dropout=0.05):
    # out = SpatialDropout2D(dropout)(out)
    out = BatchNormalization()(input_layer)
    out = ReLU()(out)
    # out = cbam_block(out, ratio=8)

    return out


def inception_module(input, dropout, num_filters, do_res=False):
    tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_1 = common_shit(tower_1, dropout)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
    tower_1 = common_shit(tower_1, dropout)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_2 = common_shit(tower_2, dropout)
    tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
    tower_2 = common_shit(tower_2, dropout)

    tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3, dropout)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
    # tower_3 = common_shit(tower_3, dropout)
    ######
    tower_3 = Conv2D(num_filters, (1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3, dropout)

    # tower_4 = Conv2D(num_filters, (5, 1), padding='same')(input)
    # tower_4 = Conv2D(num_filters, (1, 5), padding='same')(tower_4)
    # tower_4 = common_shit(tower_4, dropout)

    output = concatenate([tower_1, tower_2, tower_3], axis=3)

    if do_res:
        output = add([input, output])

    return output


def magic_inception(num_filters_first_conv, dropout, num_classes,
                    do_res=False):  # num filters conv = 270 ist goot
    m1 = Input(shape=(67, 68, 1), name='m1')
    m2 = Input(shape=(67, 68, 1), name='m2')
    input_img = concatenate([m1, m2])

    first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(5, 5), strides=(2, 2))(input_img)
    # first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(1, 3), strides=(1, 1))(first_step)
    first_step = common_shit(first_step, dropout)

    first_step = MaxPooling2D(pool_size=(2, 2))(first_step)

    inc_out_1 = inception_module(first_step, dropout, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
    link1 = MaxPooling2D(pool_size=(2, 2))(inc_out_1)
    inc_out_2 = inception_module(link1, dropout, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
    link2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inc_out_2)
    # inc_out_3 = inception_module(link2, dropout, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
    # link3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inc_out_3)

    last = Conv2D(filters=int(num_filters_first_conv * 2), kernel_size=(1, 1))(link2)
    last = common_shit(last, dropout)

    last = GlobalAveragePooling2D()(last)
    # last = Conv2D(filters=int(num_filters_first_conv / 2), kernel_size=(3, 3))(last)
    # last = common_shit(last, dropout)

    # last = PReLU()(last)
    # last = MaxPooling2D(pool_size=(2, 2))(last)
    # last = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(last)
    # last = PReLU()(last)
    # last = Flatten()(last)
    # last = GlobalAveragePooling2D()(last)
    # last = Dropout(rate=0.4)(last)
    # last = Dense(256)(last)
    # last = Dropout(0.8)(last)
    last = Dense(num_filters_first_conv)(last)
    last = LeakyReLU()(last)
    last = Dropout(0.3)(last)
    last = Dense(int(num_filters_first_conv / 3))(last)
    last = LeakyReLU()(last)
    # last = Dropout(0.5)(last)

    out = Dense(num_classes, activation='linear')(last)
    cnn = Model(inputs=[m1, m2], outputs=out)

    return cnn

# %%
# input_shape = (67, 68, 2)
#
# num_filt = 136
# model = magic_inception(num_filt, num_classes=1, dropout=0, do_res=False)
# model.compile(optimizer='adam', loss='mse')
#
# model.summary()
