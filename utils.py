import itertools
import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN


def load_magic_data():
    print('loading data')
    with open('pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    with open('pickle_data/energy_train.pkl', 'rb') as f:
        raw_energy_train = pickle.load(f)

    # y_train = raw_energy_train
    y_train = np.log10(raw_energy_train).values

    with open('pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('pickle_data/energy_test.pkl', 'rb') as f:
        raw_energy_test = pickle.load(f)

    # y_test = raw_energy_test
    y_test = np.log10(raw_energy_test).values

    print('Data dimensions:')
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # %
    batch_size = 256

    # input image dimensions
    img_rows, img_cols = 67, 68

    # %
    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, y_train, x_test, y_test, input_shape


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def std_error_log(y_true, y_pred):
    relativ_err = tf.divide((y_true - y_pred), y_true)
    return tf.keras.backend.std(relativ_err)


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


def train_adam_sgd(model, x_train, y_train, x_test, y_test, log_dir_tensorboard, net_name, initial_lr=0.001, epochs=100,
                   batch_size=350):
    # print(f'The dimension of y_tests are: {y_test.shape}, its first two elements are: {y_test[:2]}')
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=initial_lr),
                  metrics=[std_error, mean_error])

    tensorboard = TensorBoard(log_dir=log_dir_tensorboard)
    early_stop = EarlyStopping(patience=8)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('checkpoints/energy_regressor_' + net_name + '.hdf5')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=4, min_lr=0.0001)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, early_stop, check, nan_stop, reduce_lr])

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='sgd',
                  metrics=[std_error, mean_error])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, early_stop, check, nan_stop, reduce_lr])

    y_pred = model.predict(x_test)
    std_err = std_error(y_test, y_pred)
    print(f'stderr=  {std_err}')
    loss = model.evaluate(x_test, y_test)

    print('Plotting stuff...')
    plot_stuff(model, x_test, y_test, net_name)

    return loss, std_err


def train_adam(model, x_train, y_train, x_test, y_test, log_dir_tensorboard, net_name, custom_loss, initial_lr=0.001,
               epochs=100, batch_size=350):
    # print(f'The dimension of y_tests are: {y_test.shape}, its first two elements are: {y_test[:2]}')
    model.compile(loss=custom_loss,
                  optimizer=keras.optimizers.Adam(lr=initial_lr),
                  metrics=[std_error, mean_error])

    tensorboard = TensorBoard(log_dir=log_dir_tensorboard)
    early_stop = EarlyStopping(patience=9, min_delta=0.0001)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('checkpoints/energy_regressor_' + net_name + '.hdf5')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=4, min_lr=0.000005)

    result = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, early_stop, check, nan_stop, reduce_lr])

    validation_loss = np.amin(result.history['val_loss'])
    validation_std = np.amin(result.history['val_std_error'])


    print('Plotting stuff...')
    plot_stuff(model, x_test, y_test, net_name)

    return validation_loss, validation_std

def plot_stuff(model, x_test, y_test, net_name):
    sns.set()
    y_pred = model.predict(x_test)
    norm_gaus = np.divide((y_pred.flatten() - y_test), y_test)
    STD = np.std(norm_gaus)
    print(f'The STD for the shallow model is: {STD}')

    if not np.isnan(np.std(norm_gaus)):
        plt.figure()
        sns.jointplot(x=y_test, y=y_pred.flatten(), kind='hex').set_axis_labels(xlabel='True Energy',
                                                                                ylabel='Predicted Energy')
        plt.savefig('scatter_FIGO_' + net_name + '.jpg')

        plt.figure()
        sns.distplot(norm_gaus, bins=500)
        plt.title('Normalized error')
        plt.xlabel('Relative Error')
        plt.legend([net_name + ' STD=' + str(STD)])
        plt.savefig('error_distribution_' + net_name + '.jpg')
