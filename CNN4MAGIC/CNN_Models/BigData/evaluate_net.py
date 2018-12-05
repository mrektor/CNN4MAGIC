import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.loader import load_data_test
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpointsEnergyRegressorStereoTime.hdf5'
net_name = 'EnergyStereoTimeV1'
model = load_model(path)
m1_te, m2_te, energy_te = load_data_test()
y_test = np.log10(energy_te)

print('Making Predictions...')
y_pred = model.predict({'m1': m1_te, 'm2': m2_te})

# %%
print('Plotting stuff...')
plot_hist2D(y_test, y_pred, fig_folder='/data/mariotti_data/CNN4MAGIC/pics/', net_name=net_name, num_bins=100)

# %%
plot_gaussian_error(y_test, y_pred, net_name=net_name + '_10bin', num_bins=10,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/pics/')

print('All done')

# %%


# def compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins=10, plot=True, fig_folder=''):
#     '''
#     Helper function that compute the gaussian fit statistics for a nuber of bins
#     :param y_pred: Predicted y (Log Scale)
#     :param y_true: True y (Log scale)
#     :param num_bin: Number of bins
#     :return: bins_mu, bins_sigma, bins_mean_value
#     '''
#     gaussian = GaussianMixture(n_components=1)
#     bins = np.linspace(1, max(y_true), num_bins)
#
#     bins_mu = np.zeros(num_bins - 1)
#     bins_sigma = np.zeros(num_bins - 1)
#     bins_median_value = np.zeros(num_bins - 1)
#
#     if plot:
#         n_row = int(np.sqrt(num_bins - 1))
#         n_col = np.ceil((num_bins - 1) / n_row)
#         # axs, fig = plt.subplots(n_row, n_col)
#         plt.figure(figsize=(15, 15))
#
#     for i in range(num_bins - 1):
#         idx_bin = np.logical_and(y_true > bins[i], y_true < bins[i + 1])
#         y_bin_true_lin = np.power(10, y_true[idx_bin])
#         y_bin_pred_lin = np.power(10, y_pred[idx_bin].flatten())
#         error = np.divide((y_bin_true_lin - y_bin_pred_lin), y_bin_true_lin)
#         error = error[:, np.newaxis]  # Add a new axis just for interface with Gaussian Mixture
#
#         gaussian.fit(error)
#         mu = gaussian.means_
#         sigma = np.sqrt(gaussian.covariances_)
#         bins_mu[i] = mu
#         bins_sigma[i] = sigma
#         bins_median_value[i] = np.sqrt([bins[i] * bins[i + 1]])
#         if plot:
#             plt.subplot(n_row, n_col, i + 1)
#             plt.hist(error.flatten(), bins=80, density=True)
#             mu = mu.flatten()
#             sigma = sigma.flatten()
#             x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
#             plt.plot(x, norm.pdf(x, mu, sigma))
#             plt.title('Median Value: ' + "{:.2E}".format(Decimal(np.power(10, bins_median_value[i]))))
#             plt.legend(['Fitted Gaussian', 'Histogram of Error'])
#
#     if plot:
#         plt.tight_layout()
#         plt.savefig(fig_folder + net_name + '_GaussianErrorDist.png')
#         plt.savefig(fig_folder + net_name + '_GaussianErrorDist.eps')
#
#     bins_median_value_lin = np.power(10, bins_median_value)  # Bins back to linear
#     return bins_mu, bins_sigma, bins_median_value_lin
#
#
# def plot_gaussian_error(y_true, y_pred, net_name, fig_folder, num_bins=10, **kwargs):
#     bins_mu, bins_sigma, bins_median_value = compute_bin_gaussian_error(y_true, y_pred, net_name, num_bins, fig_folder=fig_folder, **kwargs)
#     fig_width = 9
#     plt.figure(figsize=(fig_width, fig_width * 0.618))
#     plt.subplot(1, 2, 1)
#     plt.semilogx(bins_median_value, bins_mu, '-*g')
#     plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_mu), np.mean(bins_mu)], 'r--')
#     plt.grid(which='both')
#     plt.legend(['Estimated $\mu$', 'Average $\mu$'])
#     plt.xlabel('Bin mean value')
#     plt.ylabel('$\mu$ of linear prediction error')
#     plt.title('$\mu$ distribution for each bin')
#     # plt.savefig('pics/bins_mu.jpg')
#
#     plt.subplot(1, 2, 2)
#     # plt.figure()
#     plt.semilogx(bins_median_value, bins_sigma, '-o')
#     plt.semilogx([min(bins_median_value), max(bins_median_value)], [np.mean(bins_sigma), np.mean(bins_sigma)], 'r--')
#     plt.grid(which='both')
#     plt.ylabel('$\sigma$ of linear prediction error')
#     plt.xlabel('Bin median value')
#     plt.title('$\sigma$ distribution for each bin')
#     plt.legend(['Estimated $\sigma$', 'Average $\sigma$'])
#     plt.tight_layout()
#     plt.savefig(fig_folder + net_name + '.png')
#     plt.savefig(fig_folder + net_name + '.eps')
#     plt.close()


plot_gaussian_error(y_test, y_pred, net_name=net_name + '_10bin', num_bins=10,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/pics/')
