import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from InterpolateMagic import InterpolateMagic

# %% Load the data
bef = time.time()
gamma_raw = pd.read_csv('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M1/allGammaM1.txt', sep=" ",
                        header=None)
now = time.time()
print('time for loading')
print(now - bef)
# hadron = pd.read_csv('/data/data/Data/M2/20180912_M2_05075292.001_Y_SS433-e1-reg-W0.40+135.txt', sep=" ", header=None)

# % Remove the trailing zeros
gamma = gamma_raw.iloc[:, 2 + 2:1041 + 2]
energy = gamma_raw.iloc[:, 1]
position = gamma_raw.iloc[:, 2:4]
# gamma.loc[:, 'class'] = 'gamma'
# hadron = hadron.iloc[:, 1:1040]
# hadron.loc[:, 'class'] = 'hadron'

print(f'gamma shape: {gamma.shape}')  # , hadron shape: {hadron.shape}')

# %%
print(f'gamma raw shape: {gamma_raw.shape}')

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
        print(f'progress: {idx/gamma.shape[0]*100} %')
    gamma_np[idx, :, :] = interpolator.interpolate(gamma.iloc[idx], remove_last=False, plot=False)


# interpolation
bef = time.time()
for idx in range(gamma.shape[0]):
    interp_gamma(idx)
now = time.time()

idx = 1
plt.figure()
plt.imshow(gamma_np[idx])
plt.colorbar()
plt.savefig('gamma_example11.jpg')

idx = 2
plt.figure()
plt.imshow(gamma_np[idx])
plt.colorbar()
plt.savefig('gamma_example12.jpg')

idx = 3
plt.figure()
plt.imshow(gamma_np[idx])
plt.colorbar()
plt.savefig('gamma_example13.jpg')

idx = 4
plt.figure()
plt.imshow(gamma_np[idx])
plt.colorbar()
plt.savefig('gamma_example14.jpg')

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
    non_train_idx = random_idxs[int(num_el * frac):]
    validation_idx = non_train_idx[:len(non_train_idx)]
    test_idx = non_train_idx[len(non_train_idx):]
    train = ndarray[train_idx, :, :]
    validation = ndarray[validation_idx, :, :]
    test = ndarray[test_idx, :, :]
    if secondarray is not None:
        train_y = secondarray[train_idx]
        validation_y = secondarray[validation_idx]
        test_y = secondarray[test_idx]
        return train, validation, test, train_y, validation_y, test_y
    else:
        return train, validation, test


gamma_numpy_train, gamma_numpy_val, gamma_numpy_test, energy_train, energy_val, energy_test = get_train_test(gamma_np,
                                                                                                             0.5,
                                                                                                             secondarray=energy)
# hadron_numpy_train, hadron_numpy_test = get_train_test(hadron_np, 0.8)

# %% Check the dimensions
print(gamma_numpy_train.shape, gamma_numpy_test.shape)
print(energy_train.shape, energy_test.shape)
# print(hadron_numpy_train.shape, hadron_numpy_test.shape)

# %% Save All
print('Saving...')

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/gamma_energy_numpy_train.json', 'wb') as f:
    json.dump(gamma_numpy_train, f)

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/gamma_energy_numpy_test.json', 'wb') as f:
    json.dump(gamma_numpy_test, f)

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/gamma_energy_numpy_val.json', 'wb') as f:
    json.dump(gamma_numpy_val, f)

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/energy_train.json', 'wb') as f:
    json.dump(energy_train, f)

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/energy_val.json', 'wb') as f:
    json.dump(energy_val, f)

with open('/data/mariotti_data/CNN4MAGIC/pickle_data/energy_test.json', 'wb') as f:
    json.dump(energy_test, f)


# with open('pickle_data/hadron_numpy_train.pkl', 'wb') as f:
#     pickle.dump(hadron_numpy_train, f)
#
# with open('pickle_data/hadron_numpy_test.pkl', 'wb') as f:
#     pickle.dump(hadron_numpy_test, f)

print('All done')
