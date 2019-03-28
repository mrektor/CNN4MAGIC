# %%

from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %%
machine = 'towerino'

BATCH_SIZE = 64
train_gn, val_gn, test_gn, position_vect = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    machine=machine,
    clean=False,
    include_time=True)

# %%
print('Loading the Neural Network...')
model = swa_model_best_more  # .load_weights(snap_to_ensemble[1])  # load_model('/home/emariott/software_magic/output_data/checkpoints/SeDense121_position_from41_SWA_from10to19.hdf5')
# model = SEDenseNet121_position(include_time=False)
# #%
# model.load_weights(
#     '/home/emariott/software_magic/output_data/swa_models/SE-121-Position-l2-notime_2019-03-11_22-48-50_SWA.h5')
print('Weight loaded')
# %
net_name = 'SeDense121_position_trainingI_SWA_6-7-10-12-14'
# model = load_model('/home/emariott/deepmagic/output_data/checkpoints/MV2-4D-30E-l2-EnsLast9_2019-02-20_11-28-13.hdf5')
# print('start predictions...')

position_prediction_6 = model.predict_generator(test_gn, verbose=1, use_multiprocessing=False, workers=7)
#%%
import pickle

print(f'Saving predictions for {net_name}...')
with open(f'output_data/reconstructions/position_{net_name}.pkl', 'wb') as f:
    pickle.dump(position_prediction_6, f)


# %%
def compute_loss(y_pred):
    # print(len(y_pred), len(energy))
    direction_limato = position_vect[:y_pred.shape[0], :]
    mse = np.mean((direction_limato - y_pred) ** 2)
    return mse


print(f'Test loss for {net_name}: {compute_loss(position_prediction_6)}')

#%%
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    machine=machine,
    clean=False,
    include_time=True)

# %%
import pickle

with open(f'output_data/reconstructions/position_{net_name}.pkl', 'wb') as f:
    pickle.dump(position_prediction, f)

# with open(f'/home/emariott/deepmagic/output_data/reconstructions/pred_{net_name}_position.pkl', 'rb') as f:
#     position_prediction = pickle.load(f)


# %%
import pickle

with open(f'/home/emariott/deepmagic/output_data/reconstructions/position_SE-DenseNet121_pos_gold_noclean_best.pkl',
          'rb') as f:
    position_prediction = pickle.load(f)

#%%
position_te_limato = position_vect[:position_prediction.shape[0], :]
energy_te_limato = energy[:position_prediction.shape[0]]
# %%

print(position_vect.shape, position_prediction.shape, position_te_limato.shape)
print(energy.shape, energy_te_limato.shape)

# %%
# import pickle
#
# with open(f'/home/emariott/deepmagic/output_data/reconstructions/pred_{net_name}_position.pkl', 'wb') as f:
#     pickle.dump(position_prediction, f)
# %%
# import pickle
# with open(f'/home/emariott/deepmagic/output_data/reconstructions/pred_{net_name}_position.pkl', 'rb') as f:
#     position_prediction = pickle.load(f)

# net_name = 'SE-DenseNet121_pos_gold_noclean_best'
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name,
                        fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction')

# %%
import numpy as np
import matplotlib.pyplot as plt

def bin_data_mask(data, num_bins, bins=None):
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), num_bins)
    binned_values = np.zeros(data.shape)
    bins_masks = []
    for i, bin in enumerate(bins):
        if i < bins.shape[0] - 1:
            mask = np.logical_and(data >= bins[i], data <= bins[i + 1])
            binned_values[mask] = bin
            bins_masks.append(mask)
    return binned_values, bins, bins_masks


def compute_theta(pos_true, pos_pred, en_bin, pos_in_mm=True, folder='', net_name='', plot=True):
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
    plt.hist(theta_sq, bins=80, log=True)
    plt.xlim([0, 0.4])
    plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
    plt.title(f'{net_name} Direction Reconstruction. Energy {en_bin}')
    plt.xlabel(r'$\theta^2$')
    plt.ylabel('Counts')
    plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
    plt.savefig(folder + '/' + net_name + '_angular_' + str(en_bin) + '.png')
    plt.savefig(folder + '/' + net_name + '_angular' + str(en_bin) + '.eps')

    return angular_resolution


def plot_angular_resolution(position_true, position_prediction, energy_true,
                            fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction',
                            net_name=''):
    binned_values, bins, bins_masks = bin_data_mask(energy_true, 11)
    resolutions = []
    bin_medians = []

    for i, mask in enumerate(bins_masks):
        bin_pos = position_true[mask]
        bin_pred_pos = position_prediction[mask]
        bin_value = np.sqrt(bins[i] * bins[i + 1])
        res = compute_theta(bin_pos, bin_pred_pos, en_bin=bin_value, plot=False,
                            folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction/histograms')
        resolutions.append(res)
        bin_medians.append(bin_value)

    state_of_the_art_theta = np.array([0.157, 0.135, 0.108, 0.095, 0.081, 0.073, 0.071, 0.067, 0.065, 0.062, 0.056])
    state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

    plt.figure()
    plt.semilogx(10 ** np.array(bin_medians), resolutions, '-o')
    plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')
    plt.xlim([100, 10000])
    plt.ylim([0, 0.175])

    plt.xlabel('Energy')
    plt.ylabel('Angular Resolution')
    plt.title('68% Containment Angular Resolution of SWA SE-DenseNet121')
    plt.legend(['Neural Network', 'MAGIC Standard Analysis'])
    plt.grid(which='both')
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.png')
    plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    # plt.show()


# %%

net_name = 'SE-DenseNet121_pos_gold_noclean_best_xlimylim'
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name,
                        fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction')
