import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %
# with open('pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
#     x_train = pickle.load(f)
print('loading...')

# with open('pickle_data/energy_train.pkl', 'rb') as f:
#     y_train = pickle.load(f)
#
# print('loading x...')
# with open('pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
#     x_test = pickle.load(f)
#
# with open('pickle_data/hadron_numpy_train.pkl', 'rb') as f:
#     hadron = pickle.load(f)

with open('/data2T/mariotti_data_2/interp_from_root/MC/result_a05to35_8_821325_Y_.pkl', 'rb') as f:
    y_test = pickle.load(f)

# %%
a = y_test['M1_interp']
b = y_test['M2_interp']
for i in range(10):
    event_idx = np.random.random_integers(0, 100)
    event_pix = a[event_idx, 1, :, :]
    event_time = a[event_idx, 0, :, :]
    event_pix2 = b[event_idx, 1, :, :]
    event_time2 = b[event_idx, 0, :, :]
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(event_pix)
    axs[0, 0].set_title('Pixels M1')
    axs[0, 1].imshow(event_time)
    axs[0, 1].set_title('Time M1')

    axs[1, 0].imshow(event_pix2)
    axs[1, 0].set_title('Pixels M2')
    axs[1, 1].imshow(event_time2)
    axs[1, 1].set_title('Time M2')
    fig.suptitle('Energy ' + str(y_test['energy'][event_idx]))
    # ax.colorbar()
    plt.savefig('test' + str(i) + '.png')

# %%
sns.set()
plt.figure()
sns.distplot(y_train, bins=500)
plt.xlabel('Energy value (linear scale)')
plt.title('Energy distribution')
# plt.show()
plt.savefig('./pics/gamma_linear_dist.png')

# %%
plt.close('all')
# %%
sns.set()
# plt.figure()
sns.distplot(np.log10(np.array(y_train).flatten()), bins=500)
plt.xlabel('Energy value ($Log_{10}$ scale)')
plt.title('Energy distribution')
# plt.show()
plt.savefig('./pics/gamma_log_dist.png')

# %%

fig, axs = plt.subplots(2, 2)
for ax in axs.flatten():
    ax.imshow(x_test[np.random.randint(1, 1000)])
    # ax.colorbar()
ax.show()

# %%
y_test = np.array(y_test)
# %%

# sns.set(style='whitegrid')
for i in range(10):
    idx = np.random.randint(1, x_test.shape[0])
    a = x_test[idx]
    plt.figure()
    plt.imshow(a)
    energy = y_test[idx]
    plt.title('Gamma with energy = ' + str(energy))
    plt.colorbar()
    plt.savefig('gamma_example' + str(i) + '.png')

# %%
with open('pickle_data/hadron_numpy_train.pkl', 'rb') as f:
    hadron = pickle.load(f)

# %%
for i in range(5):
    idx = np.random.randint(1, hadron.shape[0])
    a = hadron[idx]
    plt.figure()
    plt.imshow(a)
    plt.title('Hadron')
    plt.colorbar()
    plt.savefig('hadron_example' + str(i) + '.png')
