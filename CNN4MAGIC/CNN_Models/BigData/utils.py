import itertools
import pickle
from decimal import Decimal

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from matplotlib.colors import PowerNorm
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def load_magic_data(logx=False, energy_th=0):
    print('loading data')
    with open('/data/mariotti_data/pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
        x_train = pickle.load(f)

    with open('/data/mariotti_data/pickle_data/energy_train.pkl', 'rb') as f:
        raw_energy_train = pickle.load(f)

    # y_train = raw_energy_train
    y_train = np.log10(raw_energy_train).values

    with open('/data/mariotti_data/pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
        x_test = pickle.load(f)

    with open('/data/mariotti_data/pickle_data/energy_test.pkl', 'rb') as f:
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

    if logx is True:
        x_train = np.log10(x_train)
        x_test = np.log10(x_test)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    if energy_th > 0:
        # Create the mask and apply it to the train
        mask_tr = y_train >= energy_th
        y_train = y_train[mask_tr]
        x_train = x_train[mask_tr]

        # Do the same for the test
        mask_te = y_test >= energy_th
        y_test = y_test[mask_te]
        x_test = x_test[mask_te]

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
    check = ModelCheckpoint('checkpoints/energy_regressor_' + net_name + '.hdf5', period=6)
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
    early_stop = EarlyStopping(patience=15, min_delta=0.0001)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('checkpoints/grid_inc_filt_' + net_name + '.hdf5', period=5)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=4, min_lr=0.000005)

    result = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2,
                       validation_data=(x_test, y_test),
                       callbacks=[tensorboard, early_stop, nan_stop, reduce_lr, check])

    validation_loss = np.amin(result.history['val_loss'])
    validation_std = np.amin(result.history['val_std_error'])

    # print('Plotting stuff...')
    # plot_stuff(model, x_test, y_test, net_name)

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


def compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins=10, plot=True, fig_folder=''):
    '''
    Helper function that compute the gaussian fit statistics for a nuber of bins
    :param y_pred: Predicted y (Log Scale)
    :param y_true: True y (Log scale)
    :param num_bin: Number of bins
    :return: bins_mu, bins_sigma, bins_mean_value
    '''
    gaussian = GaussianMixture(n_components=1)
    bins = np.linspace(1, max(y_true), num_bins)

    bins_mu = np.zeros(num_bins - 1)
    bins_sigma = np.zeros(num_bins - 1)
    bins_median_value = np.zeros(num_bins - 1)

    if plot:
        n_row = int(np.sqrt(num_bins - 1))
        n_col = np.ceil((num_bins - 1) / n_row)
        # axs, fig = plt.subplots(n_row, n_col)
        plt.figure(figsize=(15, 15))

    for i in range(num_bins - 1):
        idx_bin = np.logical_and(y_true > bins[i], y_true < bins[i + 1])
        y_bin_true_lin = np.power(10, y_true[idx_bin])
        y_bin_pred_lin = np.power(10, y_pred[idx_bin].flatten())
        error_pure = np.divide((y_bin_true_lin - y_bin_pred_lin), y_bin_true_lin)
        error = error_pure[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture

        gaussian.fit(error)
        mu = gaussian.means_
        sigma = np.sqrt(gaussian.covariances_)
        bins_mu[i] = mu
        bins_sigma[i] = sigma
        bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])
        np.savetxt('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/errors/' + net_name + 'error_bin_' + str(
            bins_median_value[i]) + '.gz', error_pure)
        if plot:
            plt.subplot(n_row, n_col, i + 1)
            plt.hist(error.flatten(), bins=80, density=True)
            mu = mu.flatten()
            sigma = sigma.flatten()
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, norm.pdf(x, mu, sigma))
            plt.title('Median Value: ' + "{:.2E}".format(Decimal(np.power(10, bins_median_value[i]))))
            plt.legend(['Fitted Gaussian', 'Histogram of Error'])

    if plot:
        plt.tight_layout()
        plt.savefig(fig_folder + net_name + '_GaussianErrorDist.png')
        plt.savefig(fig_folder + net_name + '_GaussianErrorDist.eps')

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_mu, bins_sigma, bins_median_value_lin


def plot_gaussian_error(y_true, y_pred, net_name, fig_folder, num_bins=10, **kwargs):
    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins,
                                                                        fig_folder=fig_folder, **kwargs)
    fig_width = 9
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    plt.subplot(1, 2, 1)
    plt.semilogx(bins_median_value, bins_mu, '-*g')
    plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.grid(which='both')
    plt.legend(['Estimated $\mu$', 'Average $\mu$'])
    plt.xlabel('Bin mean value')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    plt.subplot(1, 2, 2)
    # plt.figure()
    plt.semilogx(bins_median_value, bins_sigma, '-o')
    plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(['Estimated $\sigma$', 'Average $\sigma$'])
    plt.tight_layout()
    plt.savefig(fig_folder + net_name + '.png')
    plt.savefig(fig_folder + net_name + '.eps')
    plt.close()


def plot_hist2D(y_true, y_pred, net_name, fig_folder, num_bins=10):
    plt.figure()
    plt.hist2d(x=y_true, y=y_pred.flatten(), bins=num_bins, cmap='inferno', norm=PowerNorm(0.65))
    plt.plot([1, 10], [1, 10], 'w-')
    plt.xlabel('True Energy (Log10)')
    plt.ylabel('Predicted Energy (Log10)')
    plt.colorbar()
    plt.title('Regression Performances ' + net_name)
    plt.legend(['Ideal Line'])
    plt.savefig(fig_folder + net_name + '.png')
    plt.savefig(fig_folder + net_name + '.eps')
    plt.close()


def bin_data(data, num_bins, bins=None):
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), num_bins)
    binned_values = np.zeros(data.shape)
    for i, bin in enumerate(bins):
        if i < bins.shape[0] - 1:
            mask = np.logical_and(data >= bins[i], data <= bins[i + 1])
            binned_values[mask] = bin
    return binned_values, bins
