import pickle

from keras.models import load_model

from CNN4MAGIC.Generator.evaluation_util import evaluate_energy
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_energy=True,
                                                                     want_golden=True
                                                                     )

# %%
# net_name = 'MobileNetV2_2dense_energy_snap_whole_11'
# model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)
# filepath = '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_11_2019-02-17_01-38-48-5.h5'
# print('loading weights...')
# model.load_weights(filepath)
model = load_model(
    '/home/emariott/deepmagic/output_data/checkpoints/MobileNetV2_4dense_energy_superconvergence_2019-02-19_00-08-23.hdf5')
print('Making predictions on test set...')
net_name = 'MobileNetV2_4dense_energy_superconvergence'
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
print(f'saving...')
with open(f'output_data/reconstructions/pred_{net_name}.pkl', 'wb') as f:
    pickle.dump(y_pred, f)

# with open(f'output_data/reconstructions/ensambels/pred_{net_name}_fourth_{filepath[-4]}', 'wb') as f:
#     pickle.dump(y_pred, f)
print('saved.')

# %%
evaluate_energy(energy_te, y_pred, net_name)

# %% ######
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins=10, plot=True, save_error=False, fig_folder='',
                               do_show=False):
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

        # Error sigma as collecting 68% of data
        # mu = np.sum(error_pure)
        # up = np.percentile(error_pure, 84)  # 100 - (100-68)/2
        # low = np.percentile(error_pure, 16)  # (100-68)/2
        # sigma = (up - low) / 2

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
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.png')
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.eps')
        if do_show:
            plt.show()

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_mu, bins_sigma, bins_median_value_lin


def plot_gaussian_error(y_true, y_pred, net_name, fig_folder, num_bins=10, do_show=False, **kwargs):
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
    plt.xlabel('Bin median value')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    plt.subplot(1, 2, 2)
    # plt.figure()
    plt.semilogx(bins_median_value, bins_sigma, '-*')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--o')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(['Estimated $\sigma$', 'Cutting Edge Technology'])
    plt.tight_layout()
    plt.savefig(fig_folder + '/' + net_name + '_bins.png')
    plt.savefig(fig_folder + '/' + net_name + '_bins.eps')
    if do_show:
        plt.show()


# %%
plot_gaussian_error(energy_te[:y_pred.shape[0]], y_pred, net_name + ' Gauss',
                    fig_folder='/home/emariott/deepmagic/output_data/pictures/energy_reconstruction')

# %%


# net_name = 'MobileNetV2_2dense_energy_snap_whole'
# files = glob(f'output_data/snapshots/{net_name}*2019-02-12*.h5')
#
# files = sorted(files)
# print(files[:-1])
# # %%
# print('Initializing NN...')
# model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)
#
# for idx, filepath in enumerate(files[:-1]):
#     print('loading weights...')
#     model.load_weights(filepath)
#
#     print('Making predictions on test set...')
#     y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
#     print(f'saving {idx+2}...')
#     with open(f'output_data/reconstructions/ensambels/pred_{net_name}_third_{filepath[-4]}', 'wb') as f:
#         pickle.dump(y_pred, f)
#     print('saved.')
#
# print('All done. Everything OK')
