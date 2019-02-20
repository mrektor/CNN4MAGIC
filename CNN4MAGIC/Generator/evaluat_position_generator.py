import matplotlib.pyplot as plt
import numpy as np

# %%
from CNN4MAGIC.CNN_Models.BigData.utils import bin_data_mask, compute_theta
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


# Loading the position data
# BATCH_SIZE = 512
# point_gn, position_te = load_point_generator(batch_size=BATCH_SIZE, want_position=True)
#
# # Lima i dati perché prendendoli a batch te ne perdi qualcuno
# position_te = np.array(position_te)
# position_te_limato = position_te[:len(point_gn) * BATCH_SIZE]
#
# # % Load the Model
# net_name = 'MobileNetV2_4dense_position-big-2'
# filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5'
# model = load_model(filepath)
#
# # Make the predictions
# print('Making predictions on test set...')
# position_prediction = model.predict_generator(generator=point_gn, verbose=1, use_multiprocessing=True, workers=8)
#
# # % Load the Energy so that you can make energy-binnings
# point_gn, energy_te = load_point_generator(batch_size=BATCH_SIZE, want_energy=True)
#
# energy_te = np.array(energy_te)
# energy_te_limato = energy_te[:len(point_gn) * BATCH_SIZE]
#
# # %
# import pickle
#
# with open('/home/emariott/deepmagic/CNN4MAGIC/Generator/position_predictions/pos_pred' + net_name + '.pkl', 'wb') as f:
#     pickle.dump(position_prediction, f)
# % Plot this resolution
# %
# plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name + ' POINT-LIKE',
#                         fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic')

def compute_theta(pos_true, pos_pred, bin_value=0, pos_in_mm=True, folder='', net_name='', plot=True):
    if pos_in_mm:
        pos_true = pos_true * 0.00337  # in deg
        pos_pred = pos_pred * 0.00337  # in deg

    num_events = pos_pred.shape[0]
    theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)

    hist_theta_sq, bins = np.histogram(theta_sq, bins=num_events)
    hist_theta_sq_normed = hist_theta_sq / float(num_events)
    cumsum_hist = np.cumsum(hist_theta_sq_normed)
    angular_resolution = np.sqrt(bins[np.where(cumsum_hist > 0.68)[0][0]])
    if not plot:
        return angular_resolution

    plt.figure()
    plt.hist(theta_sq, bins=1000, log=True)
    plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
    plt.title(f'{net_name} Direction Reconstruction. Energy bin: {bin_value}')
    plt.xlabel(r'$\theta^2$')
    plt.ylabel('Counts')
    # plt.xlim([0,0.1])
    plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
    plt.savefig(folder + '/' + net_name + '_angular.png')
    plt.savefig(folder + '/' + net_name + '_angular.eps')

    return angular_resolution


def plot_angular_resolution(position_true, position_prediction, energy_true,
                            fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic', net_name=''):
    binned_values, bins, bins_masks = bin_data_mask(energy_true, 11)
    resolutions = []
    bin_medians = []

    for i, mask in enumerate(bins_masks):
        bin_pos = position_true[mask]
        bin_pred_pos = position_prediction[mask]
        bin_value = np.sqrt(bins[i] * bins[i + 1])
        res = compute_theta(bin_pos, bin_pred_pos, bin_value=bin_value, plot=True, net_name=str(i),
                            folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic/histograms_pos')
        resolutions.append(res)
        bin_medians.append(bin_value)

    state_of_the_art_theta = np.array([0.129, 0.148, 0.120, 0.097, 0.083, 0.082, 0.077, 0.068, 0.061, 0.059, 0.055])
    state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

    plt.figure()
    plt.semilogx(10 ** np.array(bin_medians), resolutions, '-o')
    plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')

    plt.xlabel('Energy')
    plt.ylabel('Angular Resolution')
    plt.title('Angular Resolution of ' + net_name)
    plt.legend([net_name, 'State of the art'])
    plt.grid()
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.png')
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    plt.show()


# %%
# with open('/home/emariott/deepmagic/output_data/reconstructions/pos_predMobileNetV2_4dense_position-big-2.pkl', 'rb') as f:
#     position_prediction = pickle.load(f)
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_position

BATCH_SIZE = 128
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

model = MobileNetV2_4dense_position()
print('Loading weights...')
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_4dense_position_whole_2019-02-19_12-20-55-Best.h5')

print('start predictions...')
position_prediction = model.predict_generator(test_gn, verbose=1)

train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

position_te_limato = position[:position_prediction.shape[0], :]
energy_te_limato = energy[:position_prediction.shape[0]]
# %
net_name = 'MobileNetV2-4D 7 epochs best'

print(position.shape, position_prediction.shape, position_te_limato.shape)
print(energy.shape, energy_te_limato.shape)
# %%

plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name)
