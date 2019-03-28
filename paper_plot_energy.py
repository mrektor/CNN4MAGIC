import pickle
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from CNN4MAGIC.CNN_Models.BigData.utils import compute_bin_gaussian_error
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


def pkl_load(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def compute_loss_mean_absolute_linear_error(y_pred):
    energy_limato = energy_te.flatten()[:len(y_pred)]
    y_lin = np.power(10, y_pred).flatten()
    energy_limato_lin = np.power(10, energy_limato).flatten()
    error = (y_lin - energy_limato_lin) / energy_limato_lin
    mean_absolute_linear_error = np.mean(np.abs(error))
    return mean_absolute_linear_error


# %% Load True

BATCH_SIZE = 128
machine = 'titanx'

train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    include_time=True,
    clean=False)

# %% Load Reconstruction
energy_reco_filepath = []
networks = []

energy_reco_filepath.append(
    'output_data/reconstructions/SE_InceptionV3_SingleDense_energy_yesTime_from40_2019-03-17_15-35-17.pkl')
networks.append('Minimum Validation')

energy_reco_filepath.append(
    'output_data/reconstructions/SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09.pkl')
networks.append('SWA of last 10 Snapshots')

energy_reco_filepath.append(
    '/home/emariott/software_magic/output_data/reconstructions/transfer-SE-inc-v3-snap_2019-03-19_10-57-34.pkl')
networks.append('TSE-SWA (low LR)')

energy_reco_filepath.append(
    'output_data/reconstructions/transfer-SE-inc-v3-snap-LR_0_05HIGH_2019-03-20_01-50-12.pkl')
networks.append('TSE-SWA (high LR)')

energy_reco_filepath.append(
    'output_data/reconstructions/energy_transfer-SE-inc-v3-snap-LR_0_05HIGH_Best.pkl')
networks.append('TSE-SWA Minimum Validation (low LR)')

# %%
reconstructions = [pkl_load(reco_path) for reco_path in energy_reco_filepath]
losses_list = [compute_loss_mean_absolute_linear_error(reco) for reco in reconstructions]
print(losses_list)
print(networks)


# %%

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
        n_row = 2  # int(np.sqrt(num_bins - 1))
        n_col = 4  # np.ceil((num_bins - 1) / n_row)
        # axs, fig = plt.subplots(n_row, n_col)
        plt.figure(figsize=(15, 15))

    for i in range(1, num_bins - 1):
        idx_bin = np.logical_and(y_true > bins[i], y_true < bins[i + 1])
        y_bin_true_lin = np.power(10, y_true[idx_bin])
        y_bin_pred_lin = np.power(10, y_pred[idx_bin].flatten())
        error_pure = np.divide((y_bin_pred_lin - y_bin_true_lin), y_bin_true_lin)
        error_subset = error_pure[
            np.logical_and(error_pure < np.percentile(error_pure, 95), error_pure > np.percentile(error_pure, 5))]
        error = error_subset[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture

        # gaussian.fit(error)
        # mu = gaussian.means_
        # sigma = np.sqrt(gaussian.covariances_)

        # Error sigma as collecting 68% of data
        mu = np.percentile(error_pure, 50)
        up = np.percentile(error_pure, 84)  # 100 - (100-68)/2
        low = np.percentile(error_pure, 16)  # (100-68)/2
        sigma = (up - low) / 2

        bins_mu[i] = mu
        bins_sigma[i] = sigma
        bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])
        if save_error:
            np.savetxt('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/errors/' + net_name + 'error_bin_' + str(
                bins_median_value[i]) + '.gz', error_pure)
        if plot:
            plt.subplot(n_row, n_col, i)
            plt.hist(error_pure, bins=140, density=True, alpha=0.9)
            # sns.distplot(error_pure, kde=True, rug=True, bins=50)
            mu = mu.flatten()
            sigma = sigma.flatten()
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, norm.pdf(x, mu, sigma))
            plt.xlim([-1, 2.5])
            plt.title('$E_{med}$: ' + "{:.2E}".format(Decimal(np.power(10, bins_median_value[i]))))
            plt.legend(['Fitted Gaussian'])

    if plot:
        plt.tight_layout()
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.png')
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.eps')
        if do_show:
            plt.show()

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_mu, bins_sigma, bins_median_value_lin


def compute_bin_rmse(y_true, y_pred, net_name, num_bins=10, plot=False, save_error=False, fig_folder='',
                     do_show=False):
    '''
    Helper function that compute the gaussian fit statistics for a nuber of bins
    :param y_pred: Predicted y (Log Scale)
    :param y_true: True y (Log scale)
    :param num_bin: Number of bins
    :return: bins_mu, bins_sigma, bins_mean_value
    '''

    # gaussian = GaussianMixture(n_components=1)
    bins = np.linspace(1, max(y_true), num_bins)

    # bins_mu = np.zeros(num_bins - 1)
    bins_rmse = np.zeros(num_bins - 1)
    bins_median_value = np.zeros(num_bins - 1)

    if plot:
        n_row = 2  # int(np.sqrt(num_bins - 1))
        n_col = 4  # np.ceil((num_bins - 1) / n_row)
        # axs, fig = plt.subplots(n_row, n_col)
        plt.figure(figsize=(15, 15))

    for i in range(1, num_bins - 1):
        idx_bin = np.logical_and(y_true > bins[i], y_true < bins[i + 1])
        y_bin_true_lin = np.power(10, y_true[idx_bin])
        y_bin_pred_lin = np.power(10, y_pred[idx_bin].flatten())
        rmse = np.sqrt(np.sum((y_bin_pred_lin - y_bin_true_lin) ** 2))
        error_pure = np.divide((y_bin_pred_lin - y_bin_true_lin), y_bin_true_lin)
        error_subset = error_pure[
            np.logical_and(error_pure < np.percentile(error_pure, 95), error_pure > np.percentile(error_pure, 5))]
        error = error_subset[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture

        # gaussian.fit(error)
        # mu = gaussian.means_
        # sigma = np.sqrt(gaussian.covariances_)

        # Error sigma as collecting 68% of data
        mu = np.percentile(error_pure, 50)
        up = np.percentile(error_pure, 84)  # 100 - (100-68)/2
        low = np.percentile(error_pure, 16)  # (100-68)/2
        sigma = (up - low) / 2

        bins_mu[i] = mu
        bins_rmse[i] = rmse
        bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])

        if save_error:
            np.savetxt('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/errors/' + net_name + 'error_bin_' + str(
                bins_median_value[i]) + '.gz', error_pure)
        if plot:
            plt.subplot(n_row, n_col, i)
            plt.hist(error_pure, bins=140, density=True, alpha=0.9)
            # sns.distplot(error_pure, kde=True, rug=True, bins=50)
            mu = mu.flatten()
            sigma = sigma.flatten()
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, norm.pdf(x, mu, sigma))
            plt.xlim([-1, 2.5])
            plt.title('$E_{med}$: ' + "{:.2E}".format(Decimal(np.power(10, bins_median_value[i]))))
            plt.legend(['Fitted Gaussian'])

    if plot:
        plt.tight_layout()
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.png')
        plt.savefig(fig_folder + '/' + net_name + '_GaussianErrorDist.eps')
        if do_show:
            plt.show()

    bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
    return bins_rmse, bins_median_value_lin


def plot_gaussian_error_2(y_true, y_pred, net_name, fig_folder, legend_list, num_bins=10, do_show=False, **kwargs):
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

    fig_width = 9
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    marker_set = ['<', '>', 'v', '^', 's']
    for i, pred in enumerate(y_pred):
        bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                            plot=False,
                                                                            fig_folder=fig_folder, **kwargs)

        fig, axes = plt.subplots(1, 2)
        plt.semilogx(bins_median_value[1:], bins_mu[1:], marker=marker_set[i])
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, '--o')
    plt.grid(which='both', linestyle='--')
    plt.legend(legend_list)
    plt.xlabel('Bin mean value (GeV)')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    for i, pred in enumerate(y_pred):
        bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                            plot=False,
                                                                            fig_folder=fig_folder, **kwargs)
        plt.subplot(1, 2, 2)
        # plt.figure()
        plt.semilogx(bins_median_value[1:], bins_sigma[1:], marker=marker_set[i])
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--o')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both', linestyle='--')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Energy (GeV)')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(legend_list)
    plt.tight_layout()
    plt.savefig(f'{fig_folder}/{net_name}_bins.png')
    # plt.savefig(f'{fig_folder}/{net_name}_bins.pdf')
    plt.close()


# %%
minlen = np.min([len(pred) for pred in reconstructions])
y_pred = [pred[:minlen] for pred in reconstructions]
energy_te_limato = energy_te[:minlen]
networks.append('MAGIC Analysis, Aleksic (2015)')
legend_list = networks
# print(legend_list)
# %%
y_true = energy_te_limato
num_bins = 10
net_name = ''

fig_folder = '/home/emariott/software_magic/output_data/pictures/for_energy'

fig = plt.figure(constrained_layout=True, figsize=(13.27, 13.69))
gs = fig.add_gridspec(nrows=3, ncols=2  # )hspace=0, height_ratios=[1.618, 1]  # left=0.55, right=0.98,
                      )
f_ax3 = fig.add_subplot(gs[:2, 1])
# f_ax1.set_title('AX1')


f_ax1 = fig.add_subplot(gs[0, 0])
# f_ax2.set_title('AX2')

f_ax2 = fig.add_subplot(gs[1, 0], sharex=f_ax1)
f_ax2.set_title('$\sigma$ Improvement')

f_ax4 = fig.add_subplot(gs[2, :])
f_ax4.set_title('Mean absolute linear error on Test Set')

marker_set = ['P', 'X', 'D', 'o', 's']

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

for i, pred in enumerate(y_pred):
    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                        plot=False,
                                                                        fig_folder=fig_folder)
    bins_rmse, _ = compute_bin_rmse(y_true, pred, net_name, num_bins)

    f_ax1.semilogx(bins_median_value[1:], bins_mu[1:], marker=marker_set[i], linestyle='--')
    # f_ax2.loglog(bins_median_value[1:], bins_rmse[1:], marker=marker_set[i])
# plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
f_ax1.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, '-ok', linewidth=3)
f_ax1.grid(which='both', linestyle='--')
f_ax1.legend(legend_list)
# f_ax1.set_xlabel('Energy (GeV)')
f_ax1.set_ylabel('$\mu$ of linear prediction error')
f_ax1.set_title('$\mu$ distribution for each bin')
# plt.savefig('pics/bins_mu.jpg')

for i, pred in enumerate(y_pred):
    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                        plot=False,
                                                                        fig_folder=fig_folder)
    # plt.figure()
    f_ax2.semilogx(bins_median_value[1:], bins_sigma[1:], marker=marker_set[i], linestyle='--')

    sigma_interp = np.interp(cutting_edge_magic_bins_median, bins_median_value, bins_sigma)
    # mu_interp = np.interp(cutting_edge_magic_bins_median, bins_median_value, bins_mu)

    sigma_enhancement = 100 * (-sigma_interp + cutting_edge_magic_sigma) / cutting_edge_magic_sigma
    f_ax3.semilogx(cutting_edge_magic_bins_median, sigma_enhancement, marker=marker_set[i], linestyle='--')

f_ax3.grid(which='both', linestyle='--')
f_ax3.plot([47, 18929], [0, 0], '-k', linewidth=3)
# f_ax3.title('Enhancement of $\sigma$ of linear prediction error w.r.t Aleksic et al. (2016)')
f_ax3.set_xlabel('Energy (GeV)')
f_ax3.set_ylabel('Enhancement (%)')
f_ax3.set_title('Improvement of $\sigma$ w.r.t Aleksic (2015)')

f_ax2.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '-ok', linewidth=3)
# plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
f_ax2.grid(which='both', linestyle='--')
f_ax2.set_ylabel('$\sigma$ of linear prediction error')
f_ax2.set_xlabel('Energy (GeV)')
f_ax2.set_title('$\sigma$ distribution for each bin')
# f_ax3.legend(legend_list)
# plt.tight_layout()
# plt.savefig(f'{fig_folder}/{net_name}_bins.png')
# plt.savefig(f'{fig_folder}/{net_name}_bins.pdf')
# plt.close()


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
short_labels = ['MV', 'SWA', 'TSE-SWA lowLR', 'TSE-SWA high-LR', 'TSE-SWA MV']
f_ax4.barh(short_labels, losses_list, color=colors[:5])
f_ax4.set_xlim([0.19, 0.23])

# fig.subplots_adjust(hspace=0)
plt.savefig(f'{fig_folder}/glorious_plot.png')
plt.savefig(f'{fig_folder}/glorious_plot.pdf')
plt.close()

# %%

y_true = energy_te_limato
num_bins = 10
net_name = ''

fig_folder = '/home/emariott/software_magic/output_data/pictures/for_energy'

fig = plt.figure(constrained_layout=True, figsize=(13, 8))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.618, 1],
                      hspace=0)  # , height_ratios=[1.618, 1]  # left=0.55, right=0.98,

f_ax1 = fig.add_subplot(gs[:, 0])
# f_ax1.set_title('AX1')


f_ax2 = fig.add_subplot(gs[0, 1])
# f_ax2.set_title('AX2')

f_ax3 = fig.add_subplot(gs[1, 1], sharex=f_ax1)
f_ax3.set_title('$\sigma$ Improvement')

# f_ax4 = fig.add_subplot(gs[2, :])
# f_ax4.set_title('Mean absolute linear error on Test Set')

marker_set = ['P', 'X', 'D', 'o', 's']

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

for i, pred in enumerate(y_pred):
    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                        plot=False,
                                                                        fig_folder=fig_folder)
    bins_rmse, _ = compute_bin_rmse(y_true, pred, net_name, num_bins)

    f_ax1.semilogx(bins_median_value[1:], bins_mu[1:], marker=marker_set[i], linestyle='--')
    # f_ax2.loglog(bins_median_value[1:], bins_rmse[1:], marker=marker_set[i])
# plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
f_ax1.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, '-ok', linewidth=3)
f_ax1.grid(which='both', linestyle='--')
f_ax1.legend(legend_list)
# f_ax1.set_xlabel('Energy (GeV)')
f_ax1.set_ylabel('$\mu$ of linear prediction error')
f_ax1.set_title('$\mu$ distribution for each bin')
# plt.savefig('pics/bins_mu.jpg')

for i, pred in enumerate(y_pred):
    bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                        plot=False,
                                                                        fig_folder=fig_folder)
    # plt.figure()
    f_ax2.semilogx(bins_median_value[1:], bins_sigma[1:], marker=marker_set[i], linestyle='--')

    sigma_interp = np.interp(cutting_edge_magic_bins_median, bins_median_value, bins_sigma)
    # mu_interp = np.interp(cutting_edge_magic_bins_median, bins_median_value, bins_mu)

    sigma_enhancement = 100 * (-sigma_interp + cutting_edge_magic_sigma) / cutting_edge_magic_sigma
    f_ax3.semilogx(cutting_edge_magic_bins_median, sigma_enhancement, marker=marker_set[i], linestyle='--')

f_ax3.grid(which='both', linestyle='--')
f_ax3.plot([47, 18929], [0, 0], '-k', linewidth=3)
# f_ax3.title('Enhancement of $\sigma$ of linear prediction error w.r.t Aleksic et al. (2016)')
f_ax3.set_xlabel('Energy (GeV)')
f_ax3.set_ylabel('Enhancement (%)')
f_ax3.set_title('Improvement of $\sigma$ w.r.t Aleksic (2015)')

f_ax2.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '-ok', linewidth=3)
# plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
f_ax2.grid(which='both', linestyle='--')
f_ax2.set_ylabel('$\sigma$ of linear prediction error')
f_ax2.set_xlabel('Energy (GeV)')
f_ax2.set_title('$\sigma$ distribution for each bin')
# f_ax3.legend(legend_list)
# plt.tight_layout()
# plt.savefig(f'{fig_folder}/{net_name}_bins.png')
# plt.savefig(f'{fig_folder}/{net_name}_bins.pdf')
# plt.close()


# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# short_labels =['MV', 'SWA', 'TSE-SWA lowLR', 'TSE-SWA high-LR', 'TSE-SWA MV']
# f_ax4.barh(short_labels, losses_list, color=colors[:5])
# f_ax4.set_xlim([0.17, 0.33])

# fig.subplots_adjust(hspace=0)
plt.savefig(f'{fig_folder}/glorious_plot_3.png')
plt.savefig(f'{fig_folder}/glorious_plot_3.pdf')
plt.close()

# %%

plt.figure()
num_bins = 200
logbins = np.logspace(0, 5,
                      num_bins)
plt.hist(10 ** energy_te_limato, log=False, bins=logbins)
plt.grid(which='both', linestyle='--')
plt.xscale('log')
plt.savefig(f'{fig_folder}/energy_point_hist.png')
plt.close()
