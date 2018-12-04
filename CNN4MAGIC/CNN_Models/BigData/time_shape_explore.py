# import keras
import glob
import pickle
import time

# import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC/*.pkl')
# %%
befbef = time.time()
times = []
full_energy = np.zeros(269828)
full_interp_M1 = np.zeros((269828, 67, 68, 2))
full_interp_M2 = np.zeros((269828, 67, 68, 2))
# %%
num_items = 0
old = 0
print(f'number of files: {len(fileList)}')
print('start loading...')
for i, file in enumerate(fileList[:1]):
    print(str(i * 100 / len(fileList)) + '%')
    bef = time.time()
    with open(file, 'rb') as f:
        data = pickle.load(f)
        num_items = len(data['energy'])
        full_energy[old:old + num_items] = data['energy']
        full_interp_M1[old:old + num_items, 67, 68, 2] = data['M1_interp'].reshape(num_items, 67, 68, 2)
        # full_interp_M2[old:old + num_items, 67, 68, 2] = data['M2_interp']
        old = old + num_items
    now = time.time()
    times.append(now - bef)
nownow = time.time()

print('Number of items: ' + str(num_items))
print(f'Time for loading all the files: {nownow-befbef}')
print(f'Average time for loading one dict: {np.mean(np.array(times))}')

# %%
b = data['M1_interp'].reshape(num_items, 67, 68, 2)
print(b.shape)
# %%
pic = data['M1_interp'][1, 0, :, :]
print(pic.shape)
# %%
import matplotlib.pyplot as plt

plt.imshow(pic)
plt.savefig('reshape_test_data1.png')
# %%


# %%
sns.distplot(np.log10(np.array(full_energy)))
plt.savefig('/data2T/mariotti_data_2/interp_from_root/energies.png')
plt.show()
# %%
for i, el in enumerate(full_energy):
    if el == 0: print(i)
# %%
with open(fileList[10], 'rb') as f:
    data = pickle.load(f)
# %%
print(data.keys())
# %%
print(data['M1_interp'].shape)
