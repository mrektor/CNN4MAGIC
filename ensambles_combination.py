import glob
import pickle

import numpy as np

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %%
BATCH_SIZE = 512
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_energy=True,
                                                                     want_golden=False
                                                                     )

# %%
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]


# %%

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


prediction_filenames = glob.glob('output_data/reconstructions/ensambels/pred*')

prediction_filenames = sorted(prediction_filenames)
print(prediction_filenames)
# %%
predictions = np.array([load_pickle(name).flatten() for name in prediction_filenames])

# %%
print(predictions.shape)

# %%
geometric_mean = np.power(10, np.mean(predictions, axis=0))
print(geometric_mean.shape)

# %%
linear_mean = np.mean(np.power(10, predictions), axis=0)
print(linear_mean.shape)

# %%
