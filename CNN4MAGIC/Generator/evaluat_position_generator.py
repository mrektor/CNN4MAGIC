import numpy as np
from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 24 * 20
train_gn, val_gn, test_gn, position_te = load_data_generators(batch_size=BATCH_SIZE, want_position=True)

# %

# %
position_te = np.array(position_te)
position_te_limato = position_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(position_te.shape)
print(position_te_limato.shape)
# %%

net_name = 'MobileNetV2_4dense_position-big-halftrain'
filepath = '/data/code/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big.hdf5'
model = load_model(filepath)

# %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)

# %%
print(y_pred[:5], position_te_limato[:5])
# print(position_te_limato[:5])

# %%


from CNN4MAGIC.CNN_Models.BigData.utils import compute_theta, bin_data_mask

# %%
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)
# %%
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]


# %%


# %%
def plot_angular_resolution(position_true, position_prediction, energy_true,
                            fig_folder='/data/code/CNN4MAGIC/Generator/position_pic'):
    binned_values, bins, bins_masks = bin_data_mask(energy_true, 11)
    resolutions = []
    bin_medians = []
    for i, mask in enumerate(bins_masks):
        bin_pos = position_true[mask]
        bin_pred_pos = position_prediction[mask]
        bin_value = np.sqrt(bins[i] * bins[i + 1])
        res = compute_theta(bin_pos, bin_pred_pos, plot=False)
        resolutions.append(res)
        bin_medians.append(bin_value)

        plt.figure()
        plt.plot(bin_medians, resolutions)
        plt.xlabel('Energy')
        plt.ylabel('Angular Resolution')
        plt.title('Angular Resolution of ' + net_name)
        plt.grid()
        plt.savefig(fig_folder + '/angular_resolution' + net_name + '.png')
        plt.show()
