import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN


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
    y_true = tf.pow(10.0, y_true)
    y_pred = tf.pow(10.0, y_pred)
    relativ_err = tf.divide((y_true - y_pred), y_true)
    return tf.keras.backend.std(relativ_err)


def train_adam_sgd(model, x_train, y_train, x_test, y_test, log_dir_tensorboard, net_name, epochs=100, batch_size=350):
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=[std_error, std_error_log])

    tensorboard = TensorBoard(log_dir=log_dir_tensorboard)
    early_stop = EarlyStopping(patience=8)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('checkpoints/energy_regressor_' + net_name + '.hdf5')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=4, min_lr=0.0001)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, early_stop, check, nan_stop])

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='sgd',
                  metrics=[std_error, std_error_log])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard, early_stop, check, nan_stop])

    y_pred = model.predict(x_test)
    std_err_log = std_error_log(y_test, y_pred)
    std_err = std_error(y_test, y_pred)
    loss = model.evaluate(x_test, y_test)

    plot_stuff(model, x_test, y_test, net_name)

    return loss, std_err, std_error_log


def plot_stuff(model, x_test, y_test, net_name):
    sns.set()
    y_pred = model.predict(x_test)
    norm_gaus = np.divide((y_pred.flatten() - y_test), y_test)
    print(f'The STD for the shallow model is: {np.std(norm_gaus)}')

    plt.figure()
    sns.jointplot(x=y_test, y=y_pred.flatten(), kind='hex').set_axis_labels(xlabel='True Energy',
                                                                            ylabel='Predicted Energy')
    plt.savefig('scatter_FIGO_' + net_name + '.jpg')

    plt.figure()
    sns.distplot(norm_gaus, bins=500)
    plt.title('Normalized error')
    plt.xlabel('Relative Error')
    plt.legend([net_name])
    plt.savefig('error_distribution_' + net_name + '.jpg')
