import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.utils import compute_bin_gaussian_error

models = []
legends = []
m1_te, m2_te, energy_te = load_data_append('test', prune=True)

fig_width = 9
fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_width * 0.618))

for path in models:
    model = load_model(path)
    y_test = np.log10(energy_te)


def plot_gaussian_error_multiple(y_true, y_pred, net_name, fig_folder, axs, num_bins=10, first=True, last=False,
                                 legend=[], **kwargs):
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
    mu_ax = axs[0]
    sig_ax = axs[1]
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    if first:
        mu_ax.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, 'r-o')

    mu_ax.semilogx(bins_median_value, bins_mu, '-*g')

    # plt.savefig('pics/bins_mu.jpg')

    # plt.figure()
    if first:
        sig_ax.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--*')
        # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
        sig_ax.grid(which='both')
        sig_ax.ylabel('$\sigma$ of linear prediction error')
        sig_ax.xlabel('Bin median value')
        sig_ax.title('$\sigma$ distribution for each bin')
        sig_ax.legend(['Estimated $\sigma$', 'Cutting Edge Technology'])

    sig_ax.semilogx(bins_median_value, bins_sigma, '-*')

    if last:
        mu_ax.grid(which='both')
        mu_ax.legend(['Estimated $\mu$', 'Cutting Edge Technology'])
        mu_ax.xlabel('Bin mean value')
        mu_ax.ylabel('$\mu$ of linear prediction error')
        mu_ax.title('$\mu$ distribution for each bin')

        sig_ax.tight_layout()
        plt.savefig(fig_folder + net_name + '.png')
        plt.savefig(fig_folder + net_name + '.eps')
        plt.show()

    return [mu_ax, sig_ax]
