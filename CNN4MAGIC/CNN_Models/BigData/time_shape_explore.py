# import keras
import glob
import pickle
import random
import time

# import matplotlib.pyplot as plt
import numpy as np

# %%
fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last/*.pkl')
random.seed(42)
random.shuffle(fileList)

# %%
befbef = time.time()
times = []
full_energy = np.zeros(269828)
full_interp_M1 = np.zeros((269828, 67, 68, 2))
full_interp_M2 = np.zeros((269828, 67, 68, 2))

# %%
num_items = 0
old = 0
tot = 0
print(f'number of files: {len(fileList)}')
print('start loading...')
for i, file in enumerate(fileList[2250:]):
    print(str(i * 100 / len(fileList[2250:])) + '%')
    bef = time.time()
    with open(file, 'rb') as f:
        data = pickle.load(f)
        num_items = len(data['energy'])
        tot += num_items
        full_energy[old:old + num_items] = data['energy']
        # full_interp_M1[old:old + num_items, :, :, :] = data['M1_interp']
        # full_interp_M2[old:old + num_items, :, :, :] = data['M2_interp']
        old = old + num_items
    now = time.time()
    times.append(now - bef)
nownow = time.time()

print('Number of items: ' + str(tot))
print(f'Time for loading all the files: {nownow-befbef}')
print(f'Average time for loading one dict: {np.mean(np.array(times))}')

# %% Split train/validation/test
import random

idxs = [i for i in range(269828)]
random.shuffle(idxs)
idx_tr = idxs[:int(len(idxs) * 0.5)]
idx_va = idxs[int(len(idxs) * 0.5):int(len(idxs) * 0.75)]
idx_te = idxs[int(len(idxs) * 0.75):]

# %% Dump M1
print('Saving M1...')

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m1_train.pkl',
          'wb') as f:
    pickle.dump(full_interp_M1[idx_tr, :, :, :], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m1_val.pkl',
          'wb') as f:
    pickle.dump(full_interp_M1[idx_va, :, :, :], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m1_test.pkl',
          'wb') as f:
    pickle.dump(full_interp_M1[idx_te, :, :, :], f, protocol=4)

# %% Dump M2
print('Saving M2...')

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m2_train.pkl',
          'wb') as f:
    pickle.dump(full_interp_M2[idx_tr, :, :, :], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m2_val.pkl',
          'wb') as f:
    pickle.dump(full_interp_M2[idx_va, :, :, :], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/m2_test.pkl',
          'wb') as f:
    pickle.dump(full_interp_M2[idx_te, :, :, :], f, protocol=4)

# %% Dump Energy
print('Saving Energy...')

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/energy_train.pkl',
          'wb') as f:
    pickle.dump(full_energy[idx_tr], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/energy_val.pkl',
          'wb') as f:
    pickle.dump(full_energy[idx_va], f, protocol=4)

with open('/data2T/mariotti_data_2/interp_from_root/MC_whole_channel_last/energy_test.pkl',
          'wb') as f:
    pickle.dump(full_energy[idx_te], f, protocol=4)

# %%
print('All done.')
