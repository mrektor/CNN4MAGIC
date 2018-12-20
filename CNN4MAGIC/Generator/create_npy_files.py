import pickle as pkl

import numpy as np


# %%

def ROOT_dump_npy(filename, event_idx_list=None, labels=None, dump_folder='/data2T/mariotti_data_2/npy_dump/'):
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    m1 = data['M1_interp']
    m2 = data['M2_interp']

    name = filename[-40:-4]

    if event_idx_list is None:
        event_idx_list = []
    if labels is None:
        labels = {}

    for idx in range(m1.shape[0]):
        event_id_string = name + '_idx_' + str(idx)
        event_idx_list.append(event_id_string)
        evt1 = m1[idx]
        evt2 = m2[idx]

        # Update Energy, labels and position
        labels[event_id_string] = 0

        # print(evt1.shape, evt2.shape)
        b = np.zeros((67, 68, 4))
        b[:, :, :2] = evt1
        b[:, :, 2:4] = evt2
        np.save(dump_folder + '/SS433/' + event_id_string + '.npy', b)

    return event_idx_list, labels

# %% Toy test
def MC_dump_npy(filename, event_idx_list=None, labels=None, energy_labels=None, position_labels=None,
                dump_folder='/data2T/mariotti_data_2/npy_dump/'):
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    corsika = data['corsika_event_number_1']
    m1 = data['M1_interp']
    m2 = data['M2_interp']
    energy = data['energy']
    posX1 = data['src_X1'].values
    posY1 = data['src_Y1'].values

    name = filename[-23:-4]

    if event_idx_list is None:
        event_idx_list = []
    if labels is None:
        labels = {}
    if energy_labels is None:
        energy_labels = {}
    if position_labels is None:
        position_labels = {}

    for idx in range(m1.shape[0]):
        event_id_string = name + 'corsika_' + str(corsika[idx])
        event_idx_list.append(event_id_string)
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
        np.save(dump_folder + '/MC/' + event_id_string + '.npy', b)

    return event_idx_list, labels, energy_labels, position_labels


# %% Dump all ROOT files
import glob
from tqdm import tqdm

folder = '/data2T/mariotti_data_2/interp_from_root/SS433'
file_list = glob.glob(folder + '/*.pkl')
empty_files = 0
for i, file in enumerate(tqdm(file_list)):
    if i == 0:
        root_list, root_labels = ROOT_dump_npy(file)
    else:
        try:
            root_list, root_labels = ROOT_dump_npy(file, event_idx_list=root_list, labels=root_labels)
        except EOFError:
            print(f'the following is empty: {file}')
            empty_files += 1

# %%
print(len(root_list))

# %% Dump the Root dict
with open('/data2T/mariotti_data_2/npy_dump/root_list_labels.pkl', 'wb') as f:
    pkl.dump((root_list, root_labels), f)

# %% Dup all MC files

folder = '/data2T/mariotti_data_2/interp_from_root/MC_channel_last_full'
file_list = glob.glob(folder + '/*.pkl')
empty_files = 0
for i, file in enumerate(tqdm(file_list)):
    if i == 0:
        event_idx_list, labels, energy_labels, position_labels = MC_dump_npy(file)

    else:
        try:
            event_idx_list, labels, energy_labels, position_labels = MC_dump_npy(file, event_idx_list, labels,
                                                                                 energy_labels, position_labels)
        except EOFError:
            print(f'the following is empty: {file}')
            empty_files += 1

print(len(event_idx_list))

# %% Dump the MC dict
with open('/data2T/mariotti_data_2/npy_dump/MC_list_labels_energy_position.pkl', 'wb') as f:
    pkl.dump((event_idx_list, labels, energy_labels, position_labels), f)

# %% Merge the dicts and the list
# with open('/data2T/mariotti_data_2/npy_dump/MC_list_labels_energy_position.pkl', 'rb') as f:
#     event_idx_list, labels, energy_labels, position_labels = pkl.load(f)
#
# with open('/data2T/mariotti_data_2/npy_dump/root_list_labels.pkl', 'rb') as f:
#     root_list, root_labels = pkl.load(f)


total_events_list = event_idx_list + root_list
total_labels = {}
total_labels.update(labels)
total_labels.update(root_labels)

# %%
with open('/data2T/mariotti_data_2/npy_dump/total_list_labels.pkl', 'wb') as f:
    pkl.dump((total_events_list, total_labels), f)

# %%
import random

random.shuffle(total_events_list)
# %%
num_files = len(total_events_list)
partition = {}
partition['train'] = total_events_list[:int(num_files / 2)]
partition['validation'] = total_events_list[int(num_files / 2):int(num_files * 3 / 2)]
partition['test'] = total_events_list[int(num_files * 3 / 2):]

# %%
with open('/data2T/mariotti_data_2/npy_dump/train_val_test_dict_labels_list.pkl', 'wb') as f:
    pkl.dump((partition, total_labels), f)

# %% sanity_check
# folder = '/data2T/mariotti_data_2/npy_dump/SS433'
# b = np.load(folder + '/' + event_idx_list[10] + '.npy')
# import matplotlib.pyplot as plt
#
# plt.imshow(b[:, :, 0])
# plt.savefig('/data/mariotti_data/CNN4MAGIC/generator/sanity_check_pics/test.png')
# plt.show()
# # %%
# num_events = 1
# fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))
#
# i = 0
# idx = 0
# axes[0].imshow(b[:, :, 0])  # TIME
# axes[0].set_title('M1 Time')
# # axes[i, 0].set_ylabel('Gammaness: ' + str([idx]))
# axes[1].imshow(b[:, :, 1])  # PHE
# axes[1].set_title('M1 PHE')
#
# axes[2].imshow(b[:, :, 2])  # TIME
# axes[2].set_title('M2 Time')
#
# axes[3].imshow(b[:, :, 3])  # PHE
# axes[3].set_title('M2 PHE')
# plt.savefig('/data/mariotti_data/CNN4MAGIC/generator/sanity_check_pics/test.png')
#
# plt.show()
