# %%
from keras import layers
from keras.layers import *
from keras.models import Model


# %%
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def common_shit(input_layer, dropout=0.05):
    # out = SpatialDropout2D(dropout)(out)
    out = ReLU()(input_layer)
    out = BatchNormalization(center=False, scale=False)(out)
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
        output = layers.add([input, output])

    return output


def magic_inception(input_shape, num_filters_first_conv, dropout, num_classes,
                    do_res=False):  # num filters conv = 270 ist goot
    input_img = Input(shape=input_shape)

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
    cnn = Model(inputs=input_img, outputs=out)

    return cnn

# %%
input_shape = (67, 68, 1)
num_filt = 136
dropout = 0.4
model = magic_inception(input_shape, num_filt, dropout, 10)
model.summary()
