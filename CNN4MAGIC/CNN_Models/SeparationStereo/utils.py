import gc
import glob
import itertools
import pickle
import random
import time
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.colors import PowerNorm
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def load_gammas(which='train', fileListFolder='/data2T/mariotti_data_2/interp_from_root/MC_channel_last_full',
                separation=True, prune=False, impact=False, leak=False):
    fileList = glob.glob(fileListFolder + '/*.pkl')

    if len(fileList) == 0:
        print('The directory does not contain any file to load')
        raise ValueError

    random.seed(42)
    random.shuffle(fileList)

    befbef = time.time()
    times = []

    full_energy = []
    full_interp_M1 = []
    full_interp_M2 = []

    if which == 'train':
        toLoad = fileList[:750]
        print('Loading TRAIN data')

    if which == 'val':
        toLoad = fileList[800:1000]
        print('Loading VALIDATION data')

    if which == 'test':
        toLoad = fileList[-750:]
        print('Loading TEST data')

    if which == 'debug':
        toLoad = fileList[:2]
        print('Loading DEBUG data')

    print(f'number of files: {len(toLoad)}')
    print('start loading gammas...')
    for i, file in enumerate(tqdm(toLoad)):

        bef = time.time()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            if prune:
                # Conditions
                # energy_level_max = data['complete_simulation_parameters_M1']['energy'] < 1200000
                # energy_level_min = data['complete_simulation_parameters_M1']['energy'] > 0
                if impact:
                    impact1 = data['complete_simulation_parameters_M1']['impact'] < 11000
                    impact2 = data['complete_simulation_parameters_M2']['impact'] < 11000
                    imp_condition = np.logical_and(impact2, impact1)
                intensity_ok_1 = data['extras1']['intensity'] > 100
                intensity_ok_2 = data['extras2']['intensity'] > 100
                if leak:
                    leak_ok_2 = data['extras2']['leakage1_pixel'] < 0.2
                    leak_ok_1 = data['extras1']['leakage1_pixel'] < 0.2
                    lk_condition = np.logical_and(leak_ok_2, leak_ok_1)

                # condition = np.logical_and(energy_level_max, energy_level_min)
                # condition = np.logical_and(condition, impact2)
                condition = np.logical_and(intensity_ok_2, intensity_ok_1)
                condition = np.logical_and(condition, leak_ok_1)
                condition = np.logical_and(condition, leak_ok_2)

                if impact:
                    condition = np.logical_and(condition, imp_condition)

                if leak:
                    condition = np.logical_and(condition, lk_condition)

                # Pruning
                if not separation:
                    full_energy.append(data['energy'][condition].reshape(
                        (data['energy'][condition].shape[0], 1)))  # Add one axis for ease of vstack
                full_interp_M1.append(data['M1_interp'][condition])
                full_interp_M2.append(data['M2_interp'][condition])



            else:
                if not separation:
                    full_energy.append(
                        data['energy'].reshape((data['energy'].shape[0], 1)))  # Add one axis for ease of vstack
                full_interp_M1.append(data['M1_interp'])
                full_interp_M2.append(data['M2_interp'])
        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    full_interp_M1 = np.vstack(full_interp_M1)
    gc.collect()
    full_interp_M2 = np.vstack(full_interp_M2)

    print('Number of items: ' + str(full_interp_M1.shape[0]))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')
    print('cleaning memory...')
    gc.collect()
    print('cleaned.')

    if not separation:
        full_energy = np.vstack(full_energy).flatten()
        return full_interp_M1, full_interp_M2, full_energy

    else:
        gamma_labels = np.ones((full_interp_M1.shape[0], 1))
        return full_interp_M1, full_interp_M2, gamma_labels


def load_hadrons(which='train', fileListFolder='/data2T/mariotti_data_2/interp_from_root/SS433',
                 separation=True, prune=False, impact=False, leak=False):
    fileList = glob.glob(fileListFolder + '/*.pkl')

    if len(fileList) == 0:
        print('The directory does not contain any file to load')
        raise ValueError

    random.seed(42)
    random.shuffle(fileList)

    befbef = time.time()
    times = []

    full_interp_M1 = []
    full_interp_M2 = []

    if which == 'train':
        toLoad = fileList[:9]
        print('Loading TRAIN data')

    if which == 'val':
        toLoad = fileList[9:12]
        print('Loading VALIDATION data')

    if which == 'test':
        toLoad = fileList[12:22]
        print('Loading TEST data')

    if which == 'debug':
        toLoad = fileList[-2:]
        print('Loading DEBUG data')

    empty_files = 0
    print(f'number of files: {len(toLoad)}')
    print('start loading Hadrons...')
    for i, file in enumerate(tqdm(toLoad)):
        bef = time.time()
        try:
            with open(file, 'rb') as f:
                # print(f'opening {file}')
                data = pickle.load(f)

                full_interp_M1.append(data['M1_interp'])
                full_interp_M2.append(data['M2_interp'])
        except EOFError:
            empty_files += 1

        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    full_interp_M1 = np.vstack(full_interp_M1)
    gc.collect()
    full_interp_M2 = np.vstack(full_interp_M2)
    gc.collect()

    hadron_labels = np.zeros((full_interp_M1.shape[0], 1))

    print('Number of items: ' + str(hadron_labels.shape[0]))
    print('Number of empty files: ' + str(empty_files))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')
    print('cleaning memory...')
    print('cleaned.')

    return full_interp_M1, full_interp_M2, hadron_labels


def load_separation_data(which='train'):
    full_interp_M1_h, full_interp_M2_h, hadron_labels_h = load_hadrons(which)
    gc.collect()
    full_interp_M1_g, full_interp_M2_g, gamma_labels_g = load_gammas(which)
    gc.collect()
    full_M1 = np.vstack((full_interp_M1_g, full_interp_M1_h))
    del full_interp_M1_g, full_interp_M1_h
    gc.collect()
    full_M2 = np.vstack((full_interp_M2_g, full_interp_M2_h))
    del full_interp_M2_g, full_interp_M2_h
    gc.collect()
    full_labels = np.vstack((gamma_labels_g, hadron_labels_h))

    return full_M1, full_M2, full_labels


# %%

def plot_confusion_matrix(y_pred, y_test, classes,
                          normalize=False,
                          title='Confusion matrix',
                          net_name='',
                          fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' Net: ' + net_name)
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
    plt.savefig(fig_folder + '/confusion_matrix_' + net_name + '.png')
    plt.savefig(fig_folder + '/confusion_matrix_' + net_name + '.eps')
    plt.show()


def plot_gammaness(y_pred, y_true, net_name='', bins=85, fig_folder=''):
    hadrons = y_pred[y_true == 0]
    gammas = y_pred[y_true == 1]
    # sns.set()
    plt.figure()
    plt.hist(hadrons, bins=bins, log=True, histtype='step')
    plt.hist(gammas, bins=bins, log=True, histtype='step')
    plt.xlim([0, 1])
    plt.legend(['Hadrons', 'Gammas'])
    plt.title(net_name)
    plt.xlabel('Gammaness')
    plt.savefig(fig_folder + '/gammaness_' + net_name + '.png')
    plt.savefig(fig_folder + '/gammaness_' + net_name + '.eps')
    plt.show()


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


def compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins=10, plot=True, save_error=False, fig_folder=''):
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
        error_pure = np.divide((y_bin_pred_lin - y_bin_true_lin), y_bin_true_lin)
        error_subset = error_pure[
            np.logical_and(error_pure < np.percentile(error_pure, 95), error_pure > np.percentile(error_pure, 5))]
        error = error_subset[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture

        gaussian.fit(error)
        mu = gaussian.means_
        sigma = np.sqrt(gaussian.covariances_)
        bins_mu[i] = mu
        bins_sigma[i] = sigma
        bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])
        if save_error:
            np.savetxt('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/errors/' + net_name + 'error_bin_' + str(
                bins_median_value[i]) + '.gz', error_pure)
        if plot:
            plt.subplot(n_row, n_col, i + 1)
            # plt.hist(error.flatten(), bins=50, density=False)
            sns.distplot(error_pure, kde=True, rug=True, bins=50)
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
        plt.show()

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_mu, bins_sigma, bins_median_value_lin


def plot_gaussian_error(y_true, y_pred, net_name, fig_folder, num_bins=10, **kwargs):
    ######## PAPER DATA
    cutting_edge_magic_bins = [[47, 75],
                               [75, 119],
                               [119, 189],
                               [189, 299],
                               [299, 475],
                               [475, 753],
                               [753, 1194],
                               [1194, 1892],
                               [1892, 2999],
                               [2999, 4754],
                               [4754, 7535],
                               [7535, 11943],
                               [11943, 18929]]
    cutting_edge_magic_bins_median = []
    for bins in cutting_edge_magic_bins:
        median = np.sqrt(bins[0] * bins[1])
        cutting_edge_magic_bins_median.append(median)
    cutting_edge_magic_bins_median = np.array(cutting_edge_magic_bins_median)

    cutting_edge_magic_bias = np.array(
        [24.6, 7.1, -0.1, -1.5, -2.2, -2.1, -1.4, -1.8, -2.3, -1.7, -2.6, -2.1, -6.7]) * 0.01
    cutting_edge_magic_sigma = np.array(
        [21.8, 19.8, 18.0, 16.8, 15.5, 14.8, 15.4, 16.1, 18.1, 19.6, 21.9, 22.7, 20.7]) * 0.01
    cutting_edge_magic_RMS = np.array([22.5, 20.9, 21.3, 20.49, 20.20, 20.21, 21.3, 21.3, 23.2, 25.1, 26.5, 26.8, 24.4])
    ########

    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins,
                                                                        fig_folder=fig_folder, **kwargs)
    fig_width = 9
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    plt.subplot(1, 2, 1)
    plt.semilogx(bins_median_value, bins_mu, '-*g')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, 'r-o')
    plt.grid(which='both')
    plt.legend(['Estimated $\mu$', 'Cutting Edge Technology'])
    plt.xlabel('Bin mean value')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    plt.subplot(1, 2, 2)
    # plt.figure()
    plt.semilogx(bins_median_value, bins_sigma, '-*')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--*')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(['Estimated $\sigma$', 'Cutting Edge Technology'])
    plt.tight_layout()
    plt.savefig(fig_folder + net_name + '.png')
    plt.savefig(fig_folder + net_name + '.eps')
    plt.show()


def plot_hist2D(y_true, y_pred, net_name, fig_folder, num_bins=10):
    plt.figure()
    plt.hist2d(x=y_true, y=y_pred.flatten(), bins=num_bins, cmap='inferno', norm=PowerNorm(0.65))
    plt.plot([1, 10], [1, 10], 'w-')
    plt.xlabel('True Energy (Log10)')
    plt.ylabel('Predicted Energy (Log10)')
    plt.colorbar()
    plt.title('Regression Performances ' + net_name)
    plt.legend(['Ideal Line'])
    plt.xlim(1.2, 4.5)
    plt.ylim(1.2, 4.5)
    plt.savefig(fig_folder + net_name + '.png')
    plt.savefig(fig_folder + net_name + '.eps')
    plt.show()
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


def plot_misclassified_hadrons(m1_te, m2_te, y_pred_h, num_events=10, net_name='',
                               fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics/'):
    misclassified_hadrons_mask = y_pred_h > 0.5
    misclassified_probability = y_pred_h[misclassified_hadrons_mask.flatten()]
    misclassified_hadrons_M1 = m1_te[misclassified_hadrons_mask.flatten()]
    misclassified_hadrons_M2 = m2_te[misclassified_hadrons_mask.flatten()]

    print(f'there are {misclassified_hadrons_M1.shape[0]} misclassified hadrons')
    if num_events > misclassified_hadrons_M1.shape[0]:
        num_events = misclassified_hadrons_M1.shape[0]
    fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))

    indexes = [i for i in range(misclassified_hadrons_M1.shape[0])]
    random.shuffle(indexes)
    for i, idx in enumerate(indexes[:num_events]):
        axes[i, 0].imshow(misclassified_hadrons_M1[idx, :, :, 0])  # TIME
        axes[i, 0].set_title('M1 Time')
        axes[i, 0].set_ylabel('Gammaness: ' + str(misclassified_probability[idx]))

        axes[i, 1].imshow(misclassified_hadrons_M1[idx, :, :, 1])  # PHE
        axes[i, 1].set_title('M1 PHE')

        axes[i, 2].imshow(misclassified_hadrons_M2[idx, :, :, 0])  # TIME
        axes[i, 2].set_title('M2 Time')

        axes[i, 3].imshow(misclassified_hadrons_M2[idx, :, :, 1])  # PHE
        axes[i, 3].set_title('M2 PHE')

    fig.suptitle('Hadrons misclassified as Gammas')
    plt.tight_layout()
    plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.png')
    plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.pdf')
    plt.show()


def plot_misclassified_gammas(m1_te, m2_te, y_pred_g, num_events=10, net_name='',
                              fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics/'):
    misclassified_gammas_mask = y_pred_g < 0.5
    misclassified_probability = y_pred_g[misclassified_gammas_mask.flatten()]

    misclassified_gammas_M1 = m1_te[misclassified_gammas_mask.flatten()]
    misclassified_gammas_M2 = m2_te[misclassified_gammas_mask.flatten()]

    print(f'there are {misclassified_gammas_M1.shape[0]} misclassified hadrons')
    if num_events > misclassified_gammas_M1.shape[0]:
        num_events = misclassified_gammas_M1.shape[0]

    fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))
    print(misclassified_gammas_M1.shape[0])
    indexes = [i for i in range(misclassified_gammas_M1.shape[0])]
    random.shuffle(indexes)
    for i, idx in enumerate(indexes[:num_events]):
        axes[i, 0].imshow(misclassified_gammas_M1[idx, :, :, 0])  # TIME
        axes[i, 0].set_title('M1 Time')
        axes[i, 0].set_ylabel('Gammaness: ' + str(misclassified_probability[idx]))
        axes[i, 1].imshow(misclassified_gammas_M1[idx, :, :, 1])  # PHE
        axes[i, 1].set_title('M1 PHE')

        axes[i, 2].imshow(misclassified_gammas_M2[idx, :, :, 0])  # TIME
        axes[i, 2].set_title('M2 Time')

        axes[i, 3].imshow(misclassified_gammas_M2[idx, :, :, 1])  # PHE
        axes[i, 3].set_title('M2 PHE')

    fig.suptitle('Gammas misclassified as Hadrons')
    plt.tight_layout()
    plt.savefig(fig_folder + net_name + 'MisclassifiedGammas.png')
    plt.savefig(fig_folder + net_name + 'MisclassifiedGammas.pdf')
    plt.show()


def plot_classification_merit_metrics(y_pred, y_true, net_name='',
                                      fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics',
                                      save=True):
    tot_gammas = np.sum(y_true)
    tot_hadrons = y_true.shape[0] - tot_gammas

    gammaness = np.linspace(0, 1, 1000)
    epsilon_gamma = []
    epsilon_hadron = []
    Q_abelardo = []
    Q_ruben = []
    significance = []
    num_hadr_gammaness = []
    num_gamma_gammaness = []
    is_gamma = y_true == 1
    is_hadron = y_true == 0

    for threshold in gammaness:
        is_predicted_as_hadron = y_pred.flatten() < threshold
        is_predicted_as_gamma = y_pred.flatten() >= threshold

        num_hadr = np.sum(np.logical_and(is_hadron, is_predicted_as_hadron))
        num_gammas = np.sum(np.logical_and(is_gamma, is_predicted_as_gamma))

        num_hadr_gammaness.append(num_hadr)
        num_gamma_gammaness.append(num_gammas)

        # Efficiency
        epsilon_gamma.append(num_gammas / tot_gammas)
        epsilon_hadron.append(num_hadr / tot_hadrons)

        # Q
        Q_abelardo.append((num_gammas / tot_gammas) / np.sqrt(num_hadr / tot_hadrons))
        Q_ruben.append(num_gammas / np.sqrt(num_hadr))

        # S
        significance.append((num_gammas - num_hadr) / np.sqrt(num_hadr))

    # %%
    # import seaborn as sns
    # sns.set()
    plt.plot(gammaness, epsilon_hadron)
    plt.plot(gammaness, epsilon_gamma)
    plt.title('Efficiency')
    plt.xlabel('Gammaness')
    plt.ylabel('$\epsilon$')
    plt.legend(['$\epsilon_{h}$', '$\epsilon_{\gamma}$'])
    plt.grid()
    if save:
        plt.savefig(fig_folder + '/' + net_name + '_efficiency.png')
        plt.savefig(fig_folder + '/' + net_name + '_efficiency.eps')
    plt.show()

    # %%
    plt.plot(gammaness, significance)
    plt.title('Significance')
    plt.ylabel('$S$')
    plt.xlabel('Gammaness')
    plt.grid()
    if save:
        plt.savefig(fig_folder + '/' + net_name + '_significance.png')
        plt.savefig(fig_folder + '/' + net_name + '_significance.eps')
    plt.show()

    # %%
    plt.plot(gammaness, Q_abelardo)
    plt.title('Quality Factor')
    plt.ylabel('$Q$')
    plt.xlabel('Gammaness')
    plt.legend(['$\dfrac{\epsilon_{\gamma}}{\sqrt{\epsilon_{h}}}$'])
    plt.grid()
    if save:
        plt.savefig(fig_folder + '/' + net_name + '_Q.png')
        plt.savefig(fig_folder + '/' + net_name + '_Q.eps')
    plt.show()
