# import matplotlib
#
# matplotlib.use('TkAgg')
# from CNN4MAGIC.Generator.models import SE_InceptionV3_SingleDense_energy
from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %
BATCH_SIZE = 256
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True, include_time=True,
    clean=False)

# %%
# Load the model
print('Loading the Neural Network...')
model = load_model(
    '/home/emariott/deepmagic/output_data/checkpoints/Tranfer_Ensemble_SE_InceptionV3_SingleDense_energy_from40_last6_nofreeze_dense64_adam4e-4.hdf5')
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/transfer-SE-inc-v3-snap-LR_0_05HIGH_2019-03-20_01-50-12-Best.h5')
net_name = 'transfer-SE-inc-v3-snap-LR_0_05HIGH_Best'

# %%
# BATCH_SIZE=256
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# print(y_pred.shape)
# %

# net_name = 'single_DenseNet_piccina_Gold'
# filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/single_DenseNet_piccina_Gold.hdf5'
# model = load_model(filepath)
#
# %%
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=8)
# %
import pickle

#%
file = f'output_data/reconstructions/energy_{net_name}.pkl'
with open(file, 'wb') as f:
    pickle.dump(y_pred, f)
#
#%
# with open(file, 'rb') as f:
#     y_pred = pickle.load(f)

# %%
# import pickle
#
# net_name = 'SE_InceptionV3_SingleDense_energy_yesTime_from60_SWA10'
# file = f'/home/emariott/deepmagic/output_data/reconstructions/SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09.pkl'
# with open(file, 'rb') as f:
#     y_pred = pickle.load(f)

# %%
# y_pred = appello['transfer ens snap HIGHLR']
# energy_te_limato = energy_te[:y_pred.shape[0]]
# net_name = ''
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

# net_name = 'transfer_ens_snap_HIGHLR_SWA'
plot_hist2D(energy_te_limato, y_pred, net_name,
            fig_folder='output_data/pictures/energy_reconstruction',
            num_bins=250)
# %%
plot_gaussian_error(energy_te_limato, y_pred,
                    net_name=net_name,
                    fig_folder='output_data/pictures/energy_reconstruction', plot=True)
# %%

filename = '/home/emariott/deepmagic/output_data/reconstructions/energy_SEDenseNet121_energy_noclean_Gold_mse_from10_best_valloss0.2329.pkl'
with open(filename, 'rb') as f:
    y_pred = pickle.load(f)
#
#
import matplotlib.pyplot as plt

#
# plt.figure()
# sns.distplot(y_pred)
# plt.savefig('/home/emariott/ypred_net_1.png')
#
# # %%
# plt.figure()
# sns.distplot(y_pred)
# sns.distplot(energy_te_limato)
# plt.xlim([0,4])
# plt.savefig('/home/emariott/ee.png')
# plt.show()
#
# # %%
# import numpy as np
#
# a = np.array(e2)
# # %%
# a = np.array([v for k, v in e2.items()])
#
# # %%
plt.figure()
plt.hist(y_pred.flatten(), bins=1000)
# plt.hist(energy_te_limato, bins=100)
plt.xlim([0,4])
plt.savefig('/home/emariott/deepmagic/output_data/pictures/energy_diffuse.png')
#%%
import numpy as np

print(np.sum(y_pred > 5))
# %%
print(np.max(y_pred))

# %%
y_pred[y_pred > 5] = 5
# %%
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm

net_name = 'transfer_snap_se_inc_v3_HighLr_best'
y_pred = y_pred[:len(energy_te_limato)]
plt.figure()
plt.hist2d(energy_te_limato, y_pred.flatten(), bins=300, cmap='inferno', norm=PowerNorm(0.55))
plt.plot([1, 10], [1, 10], 'w-')
plt.xlabel('True Energy ($\log_{10}$)')
plt.ylabel('Predicted Energy ($\log_{10}$)')
plt.colorbar()
plt.title(f'Regression Performances of Energy Estimation')
plt.legend(['Ideal Line'])
plt.xlim(1.45, 4.4)
plt.ylim(1.45, 4.4)
# plt.suptitle('Model: Transfer Snapshot Ensemble of SE-Inception V3')
plt.savefig(f'output_data/pictures/for_paper/hist_{net_name}.pdf')
plt.close()
#%%
plt.show()

#%%
from decimal import Decimal

from CNN4MAGIC.CNN_Models.BigData.utils import compute_bin_gaussian_error

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
    plt.semilogx(bins_median_value[1:], bins_mu[1:], '-*g')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.semilogx(cutting_edge_magic_bins_median[1:], cutting_edge_magic_bias[1:], 'r-o')
    plt.grid(which='both', linestyle='--')
    plt.legend(['TSE SE-Inception V3 Best', 'Aleksic et al. (2016)'])
    plt.xlabel('Bin median value (GeV)')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    plt.xlim([47, 20929])
    plt.ylim([-0.5, 0.5])
    # plt.savefig('pics/bins_mu.jpg')

    plt.subplot(1, 2, 2)
    # plt.figure()
    plt.semilogx(bins_median_value[1:], bins_sigma[1:], '-*')
    plt.semilogx(cutting_edge_magic_bins_median[1:], cutting_edge_magic_sigma[1:], '--o')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both', linestyle='--')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value (GeV)')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(['TSE SE-Inception V3 Best', 'Aleksic et al. (2016)'])
    plt.xlim([47, 20929])
    plt.ylim([0, 0.5])
    plt.suptitle(f'Error decomposition of {net_name}', fontsize=17)
    # plt.suptitle()
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(f'{fig_folder}/{net_name}_bins.png')
    plt.savefig(f'{fig_folder}/{net_name}_bins.pdf')
    # plt.savefig(fig_folder + '/' + net_name + '_bins.eps')
    if do_show:
        plt.show()


# %%
from tqdm import tqdm

net_name = 'transfer ens snap HIGHLR SWA'
for net_name, y_pred in tqdm(zip(net_names, predictions)):
    # y_pred = appello[net_name]
    # name_for_plot = net_names[i]
    energy_te_limato = energy_te[:len(y_pred)]
    plot_gaussian_error(energy_te_limato, y_pred,
                        net_name=net_name,

                        fig_folder='/home/emariott/deepmagic/output_data/pictures/for_paper/error_no_point_horizontal',
                        plot=True)

# %%

net_names = ['Best Sanpshot', 'SWA of last 10 Snapshots', 'TSE-SWA (low LR)', 'TSE-SWA (high LR)', 'TSE Best (high LR)',
             'Aleksic et al. (2016)']
predictions = [appello['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'],
               appello['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'],
               appello['transfer ens snap'],
               appello['transfer ens snap HIGHLR SWA'],
               appello['transfer ens snap HIGHLR BEST']
               ]


# %%


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

    for pred in y_pred:
        bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                            plot=False,
                                                                            fig_folder=fig_folder, **kwargs)

        plt.subplot(1, 2, 1)
        plt.semilogx(bins_median_value, bins_mu, '-*')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_bias, 'r-o')
    plt.grid(which='both')
    plt.legend(legend_list)
    plt.xlabel('Bin mean value (GeV)')
    plt.ylabel('$\mu$ of linear prediction error')
    plt.title('$\mu$ distribution for each bin')
    # plt.savefig('pics/bins_mu.jpg')

    for pred in y_pred:
        bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, pred, net_name, num_bins,
                                                                            plot=False,
                                                                            fig_folder=fig_folder, **kwargs)
        plt.subplot(1, 2, 2)
        # plt.figure()
        plt.semilogx(bins_median_value, bins_sigma, '-*')
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--o')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value (GeV)')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(legend_list)
    plt.tight_layout()
    plt.savefig(fig_folder + '/' + net_name + '_bins.png')
    plt.savefig(fig_folder + '/' + net_name + '_bins.eps')
    if do_show:
        plt.show()


# %%
some_net_names = [
    # 'Best Sanpshot',
    # 'SWA of last 10 Snapshots',
    'TSE-SWA (low LR)',
    'TSE-SWA (high LR)',
    'TSE Best (high LR)',
    'Aleksic et al. (2016)']
some_predictions = [
    # appello['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'],
    #            appello['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'],
    appello['transfer ens snap'],
    appello['transfer ens snap HIGHLR SWA'],
    appello['transfer ens snap HIGHLR BEST']
]

some_minlen = np.min([len(pred) for pred in some_predictions])
some_pred_limate = [pred[:some_minlen] for pred in some_predictions]
some_energy_te_limato = energy_te[:some_minlen]

# %%
minlen = np.min([len(pred) for pred in predictions])
pred_limate = [pred[:minlen] for pred in predictions]
energy_te_limato = energy_te[:minlen]
# %%

folder_fig = '/home/emariott/deepmagic/output_data/pictures/for_paper/some_ensembles'
to_draw_list = some_pred_limate
legend = some_net_names  # ['Snapshot epoch 10', 'Snapshot epoch 11', 'Ensamble of epoch 10 and 11', 'Current State-of-the-Art']
plot_gaussian_error_2(energy_te_limato, to_draw_list, 'Snapshot Ensamble 11 Epochs',
                      fig_folder=folder_fig, legend_list=legend, num_bins=10,
                      do_show=False)
