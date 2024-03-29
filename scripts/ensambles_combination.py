import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from CNN4MAGIC.CNN_Models.BigData.utils import compute_bin_gaussian_error, plot_gaussian_error, plot_hist2D
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

folder_fig = '/home/emariott/deepmagic/pics'


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# %
BATCH_SIZE = 512
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_energy=True,
                                                                     want_golden=False
                                                                     )

energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
prediction_filenames = glob.glob('output_data/reconstructions/ensambels/pred*')

prediction_filenames = sorted(prediction_filenames)
print(prediction_filenames)
# %
pred_list = [load_pickle(name).flatten() for name in prediction_filenames]
predictions = np.array(pred_list)

# %
losses = [np.sum(np.power(pred - energy_te_limato, 2)) for pred in pred_list]
print(losses)

# %
linear_mean = np.mean(predictions[-3:], axis=0)
print(linear_mean.shape)

# %
ensambels = [np.mean(predictions[-i:], axis=0) for i in range(1, len(pred_list))]

# %
ensambels_losses = [np.sum(np.power(pred - energy_te_limato, 2)) for pred in ensambels]
# %%
print(ensambels_losses)
print(losses)
# %%

sns.set()
plt.figure()
plt.plot(range(5, 12), losses)
plt.xlabel('Snapshot')
plt.ylabel('MSE loss')
plt.title('Test loss of the last snapshots')
plt.savefig(f'{folder_fig}/losses.pdf')

# %%
enseambles_legend = ['Best Snapshot (11)', '11 & 10', '11 & 10 & 9', '11 & 10 & 9 & 8', '11 & 10 & 9 & 8',
                     '11 & 10 & 9 & 8 & 7']
plt.figure()
plt.barh(enseambles_legend, ensambels_losses)
plt.xlabel('MSE loss evaluated on the test set')
plt.title('Losses comparison of different Ensembles')
plt.xlim([15500, 17300])
plt.tight_layout()
plt.savefig(f'{folder_fig}/ensambels_losses.eps')

# %%

plot_hist2D(energy_te_limato, pred_list[-1], 'Modified MobileNetV2 (Best Network)', fig_folder=folder_fig, num_bins=300)
plot_gaussian_error(energy_te_limato, pred_list[-1], 'Modified MobileNetV2 (Best Network)', fig_folder=folder_fig)

plot_hist2D(energy_te_limato, ensambels[2], 'Modified MobileNetV2 (Ensamble of last 2 nets)', fig_folder=folder_fig,
            num_bins=300)
plot_gaussian_error(energy_te_limato, ensambels[2], 'Modified MobileNetV2 (Ensamble of last 2 nets)',
                    fig_folder=folder_fig)


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
    plt.xlabel('Bin mean value')
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
    plt.semilogx(cutting_edge_magic_bins_median, cutting_edge_magic_sigma, '--*')
    # plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
    plt.grid(which='both')
    plt.ylabel('$\sigma$ of linear prediction error')
    plt.xlabel('Bin median value')
    plt.title('$\sigma$ distribution for each bin')
    plt.legend(legend_list)
    plt.tight_layout()
    plt.savefig(fig_folder + '/' + net_name + '_bins.png')
    plt.savefig(fig_folder + '/' + net_name + '_bins.eps')
    if do_show:
        plt.show()


# %%

to_draw_list = [predictions[-2], predictions[-1], ensambels[1]]
legend = ['Snapshot epoch 10', 'Snapshot epoch 11', 'Ensamble of epoch 10 and 11', 'Current State-of-the-Art']
plot_gaussian_error_2(energy_te_limato, to_draw_list, 'Snapshot Ensamble 11 Epochs',
                      fig_folder=folder_fig, legend_list=legend, num_bins=10,
                    do_show=False)


# %%


# %%
import seaborn as sns

sns.set()
plt.figure()
plt.barh(legend, losses, log=False)
plt.xlabel('L2 Loss (energy in linear scale)')
plt.xlim([16000, 22000])
plt.title('Losses of various models')
plt.tight_layout()
plt.savefig('output_data/pictures/energy_reconstruction/losses_seaborn_xlim.png')
plt.savefig('output_data/pictures/energy_reconstruction/losses_seaborn_xlim.eps')

plt.close()

# %%
import seaborn as sns

sns.set()
plt.figure()
plt.barh(legend, np.log10(np.array(losses)), log=False)
plt.xlabel('Log$_{10}$ of L2 Loss (energy in linear scale)')
plt.title('Losses of various models')
plt.tight_layout()
plt.savefig('output_data/pictures/energy_reconstruction/losses_seaborn_Log.png')
plt.close()
