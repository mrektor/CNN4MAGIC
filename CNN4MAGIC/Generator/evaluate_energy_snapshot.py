import numpy as np

from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import MobileNetV2_2dense_energy

# %

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

# %
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)

# %

net_name = 'MobileNetV2_2dense_energy_snap_whole_11'
filepath = '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_11_2019-02-17_01-38-48-5.h5'
model = MobileNetV2_2dense_energy()
model.load_weights(filepath)

# %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=3)

# %
print(np.sort(y_pred[:10].flatten()))
print(energy_te_limato[:10])

# %

plot_hist2D(energy_te_limato, y_pred, net_name=net_name, fig_folder='/data/code/CNN4MAGIC/Generator/energy_pic/',
            num_bins=100)
# %
plot_gaussian_error(energy_te_limato, y_pred, net_name=net_name,
                    fig_folder='/data/code/CNN4MAGIC/Generator/energy_pic/')
# %%
