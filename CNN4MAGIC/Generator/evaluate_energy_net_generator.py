import numpy as np
from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 24 * 20
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

# %%

# %%
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# %%

net_name = 'MobileNetV2_energy-900kTrain'
filepath = '/data/code/CNN4MAGIC/Generator/checkpoints/MobileNetV2_energy-900kTrain.hdf5'
model = load_model(filepath)

# %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)

# %%
y_pred[:30]

# %%
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name, fig_folder='/data/code/CNN4MAGIC/Generator/separation_generator_pic/',
            num_bins=10)
# %%
plot_gaussian_error(energy_te_limato,)
