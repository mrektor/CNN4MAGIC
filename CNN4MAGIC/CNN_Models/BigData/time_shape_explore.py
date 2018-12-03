# import keras
import glob
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC/*.pkl')
# %%
befbef = time.time()
times = []
print('start loading...')
for file in fileList:
    print(file)
    bef = time.time()
    with open(file, 'rb') as f:
        a = pickle.load(f)
    now = time.time()
    times.append(now - bef)
nownow = time.time()

print(f'Time for loading all the files: {nownow-befbef}')
print(f'Average time for loading one dict: {np.mean(np.array(times))}')

sns.distplot(np.array(times))
plt.savefig('/data2T/mariotti_data_2/interp_from_root/time_load_dist.png')
plt.show()

# %%
with open(fileList[10], 'rb') as f:
    a = pickle.load(f)
# %%
print(a.keys())
# %%
print(a['M1_interp'].shape)
