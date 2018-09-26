import pickle

import numpy as np
import pandas as pd

from InterpolateMagic import InterpolateMagic

# %% Load the data
gamma = pd.read_csv('../data/gamma/Output_pixels_M1.txt', sep=" ", header=None)
hadron = pd.read_csv('../data/proton/Output_pixels_M1.txt', sep=" ", header=None)

# % Remove the trailing zeros
gamma = gamma.iloc[:, :1039]
gamma.loc[:, 'class'] = 'gamma'
hadron = hadron.iloc[:, :1039]
hadron.loc[:, 'class'] = 'hadron'

# %% Call the interpolator
interpolator = InterpolateMagic(10)
a = interpolator.interpolate(gamma.iloc[0, :-1], remove_last=False, plot=True)  # Test

# %% Interpolate the gammas
gamma_np = np.zeros((gamma.shape[0], a.shape[0], a.shape[1]))
for idx, _ in enumerate(gamma):
    gamma_np[idx, :, :] = interpolator.interpolate(gamma.iloc[idx, :-1], remove_last=False, plot=False)

# %% Interpolate the hadron
hadron_np = np.zeros((hadron.shape[0], a.shape[0], a.shape[1]))
for idx, _ in enumerate(hadron):
    hadron_np[idx, :, :] = interpolator.interpolate(hadron.iloc[idx, :-1], remove_last=False, plot=False)

# %%

with open('../data/gamma_numpy.pkl', 'wb') as f:
    pickle.dump(gamma_np, f)

with open('../data/hadron_numpy.pkl', 'wb') as f:
    pickle.dump(hadron_np, f)
