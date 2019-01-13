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


from CNN4MAGIC.CNN_Models.BigData.utils import compute_theta

# %%
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)
# %%
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]


# %%

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


# %%
binned_values, bins, bins_masks = bin_data_mask(energy_te_limato, 11)

resolutions = []
bin_medians = []
for i, mask in enumerate(bins_masks):
    bin_pos = position_te_limato[mask]
    bin_pred_pos = y_pred[mask]
    bin_value = np.sqrt(bins[i] * bins[i + 1])
    res = compute_theta(bin_pos, bin_pred_pos,
                        folder='/data/code/CNN4MAGIC/Generator/position_pic',
                        net_name=net_name + ' energy = ' + str(bin_value))
    resolutions.append(res)
    bin_medians.append(bin_value)

# %%
import matplotlib.pyplot as plt

# %%
plt.figure()
plt.plot(bin_medians, resolutions)
plt.xlabel('Energy')
plt.ylabel('Angular Resolution')
plt.title('Angular Resolution of ' + net_name)
plt.grid()
plt.savefig('/data/code/CNN4MAGIC/Generator/position_pic/angular_resolution.png')
plt.show()
