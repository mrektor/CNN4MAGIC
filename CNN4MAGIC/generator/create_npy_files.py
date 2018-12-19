import pickle as pkl

import numpy as np

# %% Toy test
filename = '/data2T/mariotti_data_2/interp_from_root/MC_channel_last_full/result_a05to35_8_821318_Y_.pkl'
with open(filename, 'rb') as f:
    a = pkl.load(f)

# %%
print(a.keys())
# %%
corsika = a['corsika_event_number_1']

# %%
dump_folder = '/data2T/mariotti_data_2/npy_dump'

m1 = a['M1_interp']
m2 = a['M2_interp']
corsika = a['corsika_event_number_1']
energy = a['energy']
posX1 = a['src_X1'].values
posY1 = a['src_Y1'].values

# %%
print([posX1[0], posY1[0]])

# %%
name = filename[-23:-4]
print(name)
# %%
event_idx_list = []
labels = {}
energy_labels = {}
position_labels = {}
folder_dir = '/data/mariotti_data/CNN4MAGIC/generator/test_npy/'
for idx in range(m1.shape[0]):
    event_id_string = name + 'corsika_' + str(corsika[idx])
    event_idx_list.append(event_id_string)
    print(event_id_string)
    evt1 = m1[idx]
    evt2 = m2[idx]

    # Update Energy, labels and position
    labels[event_id_string] = 1
    energy_labels[event_id_string] = energy[idx]
    position_labels[event_id_string] = [posX1[idx], posY1[idx]]

    # print(evt1.shape, evt2.shape)
    b = np.zeros((67, 68, 4))
    b[:, :, :2] = evt1
    b[:, :, 2:4] = evt2
    np.save(folder_dir + event_id_string + '.npy', b)

# %%
print(position_labels)

# %% sanity_check
import matplotlib.pyplot as plt

plt.imshow(b[:, :, 0])
plt.savefig('/data/mariotti_data/CNN4MAGIC/generator/sanity_check_pics/test.png')
plt.show()
# %%
num_events = 1
fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))
# print(misclassified_gammas_M1.shape[0])
# indexes = [i for i in range(misclassified_gammas_M1.shape[0])]
# random.shuffle(indexes)
# for i, idx in enumerate(indexes[:num_events]):
i = 0
idx = 0
axes[0].imshow(b[:, :, 0])  # TIME
axes[0].set_title('M1 Time')
# axes[i, 0].set_ylabel('Gammaness: ' + str([idx]))
axes[1].imshow(b[:, :, 1])  # PHE
axes[1].set_title('M1 PHE')

axes[2].imshow(b[:, :, 2])  # TIME
axes[2].set_title('M2 Time')

axes[3].imshow(b[:, :, 3])  # PHE
axes[3].set_title('M2 PHE')
plt.savefig('/data/mariotti_data/CNN4MAGIC/generator/sanity_check_pics/test.png')

plt.show()
