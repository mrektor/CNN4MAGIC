import numpy as np
from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 641
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

# %%
print(len(energy_te))
print(len(test_gn) * BATCH_SIZE)
# %%
labels = np.array(energy_te)

# %%

net_name = 'MobileNetV2-separation-big'
filepath = '/data/code/CNN4MAGIC/Generator/checkpoints/MobileNetV2-separation-big.hdf5'
model = load_model(filepath)

# %%
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)
