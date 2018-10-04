import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# with open('pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
#     x_train = pickle.load(f)
print('loading...')

with open('pickle_data/energy_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

print('loading x...')
with open('pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
    x_test = pickle.load(f)

with open('pickle_data/energy_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# %%
sns.set()
plt.figure()
sns.distplot(y_train, bins=500)
plt.show()

# %%
import numpy as np

plt.figure()
sns.distplot(np.log(y_train), bins=500)
plt.show()

# %%
sns.set(style='whitegrid')
idx = 0
a = x_test[idx]
plt.figure()
plt.imshow(a)
plt.colorbar()
plt.show()
# plt.savefig('gamma_example1.jpg')
