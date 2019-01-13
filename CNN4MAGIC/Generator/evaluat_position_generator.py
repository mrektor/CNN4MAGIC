import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution
from CNN4MAGIC.Generator.gen_util import load_data_generators

# Loading the position data
BATCH_SIZE = 24 * 20
train_gn, val_gn, test_gn, position_te = load_data_generators(batch_size=BATCH_SIZE, want_position=True)

# Lima i dati perché prendendoli a batch te ne perdi qualcuno
position_te = np.array(position_te)
position_te_limato = position_te[:len(test_gn) * BATCH_SIZE]

# %% Load the Model
net_name = 'MobileNetV2_4dense_position-big-halftrain'
filepath = '/data/code/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big.hdf5'
model = load_model(filepath)

# Make the predictions
print('Making predictions on test set...')
position_prediction = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)

# %% Load the Energy so that you can make energy-binnings
_, _, _, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %% Plot this resolution
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name)
