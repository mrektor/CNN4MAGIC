import multiprocessing
import pickle
import time

import numpy as np
import pandas as pd

from InterpolateMagic import InterpolateMagic

# %% Load the data
gamma_raw = pd.read_csv('/data/data/MC/M1/energy_gammas.txt', sep=" ", header=None)
# hadron = pd.read_csv('/data/data/Data/M2/20180912_M2_05075292.001_Y_SS433-e1-reg-W0.40+135.txt', sep=" ", header=None)

# % Remove the trailing zeros
gamma = gamma_raw.iloc[:, 2:1041]
energy = gamma_raw.iloc[:, 1]
# gamma.loc[:, 'class'] = 'gamma'
# hadron = hadron.iloc[:, 1:1040]
# hadron.loc[:, 'class'] = 'hadron'

print(f'gamma shape: {gamma.shape}')  # , hadron shape: {hadron.shape}')

# %% Call the interpolator
interpolator = InterpolateMagic(15)

# %% Test

idx = np.random.randint(0, 10000)
print(idx)
a = interpolator.interpolate(gamma.iloc[idx], remove_last=False, plot=False)  # Test
# b = interpolator.interpolate(hadron.iloc[idx], remove_last=False, plot=True)  # Test

# plt.figure()
# plt.imshow(a)
# plt.colorbar()
# plt.savefig('gamma_example1.jpg')
# plt.figure()
# plt.imshow(b)
# plt.colorbar()
# plt.savefig('hadron_example2.jpg')
# %%

# %% Interpolate the gammas
print('begin gamma interpolation')
gamma_np = np.zeros((gamma.shape[0], a.shape[0], a.shape[1]))


def interp_gamma(idx):
    if idx % 100 == 0:
        print(f'progress: {idx/gamma.shape[0]*3000} %')
    gamma_np[idx, :, :] = interpolator.interpolate(gamma.iloc[idx], remove_last=False, plot=False)


# Parallel interpolation
bef = time.time()
pool = multiprocessing.Pool()
pool.map(interp_gamma, range(gamma.shape[0]))
pool.close()
pool.join()
now = time.time()

print(f'Time taken for the interpolation: {(now-bef)/60} minutes')
# % Interpolate the hadron
# print('begin hadron interpolation')
# hadron_np = np.zeros((hadron.shape[0], a.shape[0], a.shape[1]))
# for idx in range(hadron.shape[0]):
#     if idx % 100 == 0:
#         print(f'progress: {idx/hadron.shape[0]*100} %')
#     hadron_np[idx, :, :] = interpolator.interpolate(hadron.iloc[idx], remove_last=False, plot=False)

# %%
print(gamma_np.shape)


# print(hadron_np.shape)


# %% Split in train/test
def get_train_test(ndarray, frac, secondarray=None):
    num_el = ndarray.shape[0]
    random_idxs = np.random.permutation(num_el)
    train_idx = random_idxs[:int(num_el * frac)]
    test_idx = random_idxs[int(num_el * frac):]
    train = ndarray[train_idx, :, :]
    test = ndarray[test_idx, :, :]
    if secondarray is not None:
        train_y = secondarray[train_idx]
        test_y = secondarray[test_idx]
        return train, test, train_y, test_y
    else:
        return train, test


gamma_numpy_train, gamma_numpy_test, energy_train, energy_test = get_train_test(gamma_np, 0.8, secondarray=energy)
# hadron_numpy_train, hadron_numpy_test = get_train_test(hadron_np, 0.8)

# %% Check the dimensions
print(gamma_numpy_train.shape, gamma_numpy_test.shape)
print(energy_train.shape, energy_test.shape)
# print(hadron_numpy_train.shape, hadron_numpy_test.shape)

# %% Save All
print('Saving...')

with open('pickle_data/gamma_energy_numpy_train.pkl', 'wb') as f:
    pickle.dump(gamma_numpy_train, f, protocol=4)

with open('pickle_data/gamma_energy_numpy_test.pkl', 'wb') as f:
    pickle.dump(gamma_numpy_test, f, protocol=4)

with open('pickle_data/energy_train.pkl', 'wb') as f:
    pickle.dump(energy_train, f, protocol=4)

with open('pickle_data/energy_test.pkl', 'wb') as f:
    pickle.dump(energy_test, f, protocol=4)

# with open('pickle_data/hadron_numpy_train.pkl', 'wb') as f:
#     pickle.dump(hadron_numpy_train, f)
#
# with open('pickle_data/hadron_numpy_test.pkl', 'wb') as f:
#     pickle.dump(hadron_numpy_test, f)

print('All done')
