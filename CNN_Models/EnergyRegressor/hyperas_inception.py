import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import *
from keras.layers import *
from keras.losses import *
from keras.models import Model

from utils import load_magic_data


def common_shit(input_layer):
    if {{choice(['relu', 'leakyrelu'])}} == 'relu':
        out = ReLU()(input_layer)
    else:
        out = LeakyReLU()(input_layer)

    if {{choice(['do batch norm', 'no batch norm'])}} == 'do batch norm':
        out = BatchNormalization(center=False, scale=False)(out)

    return out


def inception_module(input, num_filters):
    tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_1 = common_shit(tower_1)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
    tower_1 = common_shit(tower_1)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_2 = common_shit(tower_2)
    tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
    tower_2 = common_shit(tower_2)

    tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = Conv2D(num_filters, (1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=3)

    if {{choice(['do res', 'no res'])}} == 'do res':  # do the residual connection
        output = add([input, output])

    return output


def magic_inception(input_shape, num_filters_first_conv, dropout, do_res=False):  # num filters conv = 270 ist goot
    input_img = Input(shape=input_shape)

    first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(5, 5), strides=(2, 2))(input_img)
    # first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(1, 3), strides=(1, 1))(first_step)
    first_step = common_shit(first_step, dropout)

    first_step = MaxPooling2D(pool_size=(2, 2))(first_step)

    inc_out_1 = inception_module(first_step, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
    link1 = MaxPooling2D(pool_size=(2, 2))(inc_out_1)
    inc_out_2 = inception_module(link1, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
    link = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inc_out_2)

    #########
    if {{choice(['three inception', 'two inception'])}} == 'three inception':  # Do 3 layers of inception
        inc_out_3 = inception_module(link, num_filters=int(num_filters_first_conv / 3), do_res=do_res)
        link = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inc_out_3)

    last = Conv2D(filters=int(num_filters_first_conv * 2), kernel_size=(1, 1))(link)
    last = common_shit(last)

    last = GlobalAveragePooling2D()(last)

    last = Dense(num_filters_first_conv)(last)

    out = Dense(1, activation='linear')(last)
    cnn = Model(inputs=input_img, outputs=out)

    return cnn


def std_error(y_true, y_pred):
    # print(y_true.shape)
    y_true = tf.pow(10.0, y_true)
    y_pred = tf.pow(10.0, y_pred)
    relativ_err = tf.divide((y_true - y_pred), y_true)
    return tf.keras.backend.std(relativ_err)


def mean_error(y_true, y_pred):
    y_true = tf.pow(10.0, y_true)
    y_pred = tf.pow(10.0, y_pred)
    return tf.add(y_true, -y_pred)


def create_model(x_train, y_train, x_test, y_test):
    # num_filt = [2 * 3 * i for i in range(5, 30)]

    def common_shit(input_layer):
        if {{choice(['relu', 'leakyrelu'])}} == 'relu':
            out = ReLU()(input_layer)
        else:
            out = LeakyReLU()(input_layer)

        if {{choice(['do batch norm', 'no batch norm'])}} == 'do batch norm':
            out = BatchNormalization(center=False, scale=False)(out)

        return out

    num_filters_first_conv = {{choice(
        [18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132,
         138, 144, 150])}}

    input_img = Input(shape=(67, 68, 1))

    first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(5, 5), strides=(2, 2))(input_img)
    # first_step = Conv2D(filters=int(num_filters_first_conv), kernel_size=(1, 3), strides=(1, 1))(first_step)
    first_step = common_shit(first_step)

    first_step = MaxPooling2D(pool_size=(2, 2))(first_step)

    num_filters = int(num_filters_first_conv / 3)

    ##### INC 1 #####
    input = first_step
    tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_1 = common_shit(tower_1)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
    tower_1 = common_shit(tower_1)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_2 = common_shit(tower_2)
    tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
    tower_2 = common_shit(tower_2)

    tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = Conv2D(num_filters, (1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=3)

    if {{choice(['do res', 'no res'])}} == 'do res':  # do the residual connection
        output = add([input, output])

    #####

    link1 = MaxPooling2D(pool_size=(2, 2))(output)

    ##### INC 2 ######

    input = link1
    tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_1 = common_shit(tower_1)
    tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
    tower_1 = common_shit(tower_1)

    tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
    tower_2 = common_shit(tower_2)
    tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
    tower_2 = common_shit(tower_2)

    tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)
    tower_3 = Conv2D(num_filters, (1, 1), padding='same')(tower_3)
    tower_3 = common_shit(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis=3)

    if {{choice(['do res', 'no res'])}} == 'do res':  # do the residual connection
        output = add([input, output])

    ##################

    link = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)

    #########
    if {{choice(['three inception', 'two inception'])}} == 'three inception':  # Do 3 layers of inception

        ################ INC 3 ########
        input = link
        tower_1 = Conv2D(num_filters, (1, 1), padding='same')(input)
        tower_1 = common_shit(tower_1)
        tower_1 = Conv2D(num_filters, (3, 3), padding='same')(tower_1)
        tower_1 = common_shit(tower_1)

        tower_2 = Conv2D(num_filters, (1, 1), padding='same')(input)
        tower_2 = common_shit(tower_2)
        tower_2 = Conv2D(num_filters, (4, 4), padding='same')(tower_2)
        tower_2 = common_shit(tower_2)

        tower_3 = Conv2D(num_filters, (3, 1), strides=(1, 1), padding='same')(input)
        tower_3 = Conv2D(num_filters, (1, 3), strides=(1, 1), padding='same')(tower_3)
        tower_3 = common_shit(tower_3)
        tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)
        tower_3 = common_shit(tower_3)
        tower_3 = Conv2D(num_filters, (1, 1), padding='same')(tower_3)
        tower_3 = common_shit(tower_3)

        output = concatenate([tower_1, tower_2, tower_3], axis=3)

        if {{choice(['do res', 'no res'])}} == 'do res':  # do the residual connection
            output = add([input, output])
        ###############################

        link = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)

    last = Conv2D(filters=int(num_filters_first_conv * 2), kernel_size=(1, 1))(link)
    last = common_shit(last)

    last = GlobalAveragePooling2D()(last)

    last = Dense(num_filters_first_conv)(last)

    out = Dense(1, activation='linear')(last)
    model = Model(inputs=input_img, outputs=out)

    tensorboard = TensorBoard(log_dir='Hyperas_tensorboard')
    early_stop = EarlyStopping(patience=10, min_delta=0.0001)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('checkpoints/hyperas.hdf5')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=4, min_lr=0.000001)

    model.compile(loss=mean_absolute_error,
                  optimizer={{choice(['sgd', 'adam', 'rmsprop'])}},
                  metrics=[std_error, mean_error])

    result = model.fit(x_train, y_train,
                       batch_size={{choice([128, 256, 350, 512])}},
                       epochs=60,
                       verbose=2,
                       validation_data=(x_test, y_test),
                       callbacks=[tensorboard, early_stop, nan_stop, reduce_lr, check])

    # get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss'])
    validation_std = np.amin(result.history['val_std_error'])

    print('Best validation loss of epoch:', validation_loss)
    print('Best validation std of epoch:', validation_std)
    print()

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


def data():
    x_train, y_train, x_test, y_test, input_shape = load_magic_data()
    print('Data Loaded')
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
