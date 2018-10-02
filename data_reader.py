import pickle

import numpy as np
import pandas as pd

from InterpolateMagic import InterpolateMagic

# %% Load the data
gamma = pd.read_csv('/data/data/MC/M1/all_gammas_MC_M1.txt', sep=" ", header=None)
hadron = pd.read_csv('/data/data/Data/M2/20180912_M2_05075292.001_Y_SS433-e1-reg-W0.40+135.txt', sep=" ", header=None)

# % Remove the trailing zeros
gamma = gamma.iloc[:, :1039]
gamma.loc[:, 'class'] = 'gamma'
hadron = hadron.iloc[:, :1039]
hadron.loc[:, 'class'] = 'hadron'

# %%
print(f'gamma shape: {gamma.shape}, hadron shape: {hadron.shape}')

# %% Call the interpolator
interpolator = InterpolateMagic(10)
a = interpolator.interpolate(gamma.iloc[0, :-1], remove_last=False, plot=True)  # Test

# %% Interpolate the gammas
print('begin gamma interpolation')
gamma_np = np.zeros((gamma.shape[0], a.shape[0], a.shape[1]))
for idx in range(gamma.shape[0]):
    if idx % 100 == 0:
        print(f'progress: {idx/gamma.shape[0]*100} %')
    gamma_np[idx, :, :] = interpolator.interpolate(gamma.iloc[idx, :-1], remove_last=False, plot=False)

# %% Interpolate the hadron
print('begin hadron interpolation')
hadron_np = np.zeros((hadron.shape[0], a.shape[0], a.shape[1]))
for idx in range(hadron.shape[0]):
    if idx % 100 == 0:
        print(f'progress: {idx/hadron.shape[0]*100} %')
    hadron_np[idx, :, :] = interpolator.interpolate(hadron.iloc[idx, :-1], remove_last=False, plot=False)

# %%
print(gamma_np.shape)
print(hadron_np.shape)


# %% Split in train/test
def get_train_test(ndarray, frac):
    num_el = ndarray.shape[0]
    random_idxs = np.random.permutation(num_el)
    train_idx = random_idxs[:int(num_el * frac)]
    test_idx = random_idxs[int(num_el * frac):]
    train = ndarray[train_idx, :, :]
    test = ndarray[test_idx, :, :]
    return train, test


gamma_numpy_train, gamma_numpy_test = get_train_test(gamma_np, 0.8)
hadron_numpy_train, hadron_numpy_test = get_train_test(hadron_np, 0.8)

# %% Check the dimensions
print(gamma_numpy_train.shape, gamma_numpy_test.shape)
print(hadron_numpy_train.shape, hadron_numpy_test.shape)

# %% Save All
print('Saving...')

with open('pickle_data/gamma_numpy_train.pkl', 'wb') as f:
    pickle.dump(gamma_numpy_train, f)

with open('pickle_data/gamma_numpy_test.pkl', 'wb') as f:
    pickle.dump(gamma_numpy_test, f)

with open('pickle_data/hadron_numpy_train.pkl', 'wb') as f:
    pickle.dump(hadron_numpy_train, f)

with open('pickle_data/hadron_numpy_test.pkl', 'wb') as f:
    pickle.dump(hadron_numpy_test, f)

print('All done')
