# %%
from keras import layers
from keras.layers import Input, BatchNormalization, ReLU, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Dense, \
    concatenate
from keras.models import Model


# %%

def common_shit(input_layer, dropout=0.05):
    # out = SpatialDropout2D(dropout)(out)
    out = ReLU()(input_layer)
    out = BatchNormalization(center=False, scale=False)(out)

    return out


def inception_module(input, dropout, num_filters, do_res=False):
    tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_1 = common_shit(tower_1, dropout)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
    tower_1 = common_shit(tower_1, dropout)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_2 = common_shit(tower_2, dropout)
    tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
    tower_1 = common_shit(tower_1, dropout)

    tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3, dropout)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3, dropout)
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


def magic_inception(input_shape, num_filters_first_conv, dropout, do_res=False):  # num filters conv = 270 ist goot
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
    last = Dense(num_filters_first_conv)(last)
    # last = Dropout(0.5)(last)

    out = Dense(1, activation='linear')(last)
    cnn = Model(inputs=input_img, outputs=out)

    return cnn

# %%
# input_shape = (67, 68, 1)
# num_filt = 2 * 3 * 40
# dropout = 0.4
# model = magic_inception(input_shape, num_filt, dropout)
# model.summary()
