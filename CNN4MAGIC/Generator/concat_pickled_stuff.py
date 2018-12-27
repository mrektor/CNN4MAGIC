import glob

from tqdm import tqdm

# %%
filelist = glob.glob('/data2T/mariotti_data_2/MC_npy/complementary_computation/*')

# %%
print(filelist[0])
# %%
import pickle

eventList_total = []
energy_total = {}
labels_total = {}
position_total = {}

for file in tqdm(filelist):
    with open(file, 'rb') as f:
        eventList, labels, energy, position = pickle.load(f)
    eventList_total = eventList_total + eventList
    energy_total.update(energy)
    labels_total.update(labels)
    position_total.update(position)

# %
print(len(eventList_total))
print(len(energy_total.keys()))
# %%
num_files = len(eventList_total)
partition = {}
frac_train = 0.67
frac_val = 0.10
partition['train'] = eventList_total[:int(num_files * frac_train)]
partition['validation'] = eventList_total[int(num_files * frac_train):int(num_files * (frac_train + frac_val))]
partition['test'] = eventList_total[int(num_files * (frac_train + frac_val)):]

# %%
print(len(partition['train']))
print(len(partition['validation']))
print(len(partition['test']))

# %%
with open('/data2T/mariotti_data_2/MC_npy/complementary_dump_total_2.pkl', 'wb') as f:
    pickle.dump((partition, energy_total, labels_total, position_total), f)

#############
# %%
to_move = glob.glob('/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/*.npy')
# %%
print(len(to_move))
# # %%
# print(to_move[0])
# # %%
# import os
#
# for idx in tqdm(range(1, len(to_move))):
#     os.system('mv ' + to_move[idx] + ' /data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/')

# %%
import random

eventList_total = glob.glob('/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/*')
newlist = []
for event in eventList_total:
    newlist.append(event[66:-4])
# %%
eventList_total = newlist
random.seed(42)
random.shuffle(eventList_total)
num_files = len(eventList_total)
partition = {}
frac_train = 0.67
frac_val = 0.10
partition['train'] = eventList_total[:int(num_files * frac_train)]
partition['validation'] = eventList_total[int(num_files * frac_train):int(num_files * (frac_train + frac_val))]
partition['test'] = eventList_total[int(num_files * (frac_train + frac_val)):]


# %%
def clean_missing_data(data, labels):
    p = 0
    todelete = []
    for key in data:
        try:
            a = labels[key]
        except KeyError:
            todelete.append(key)
            p = p + 1
    print(f'solved {len(todelete)} of KeyErrors.')
    for key in todelete:
        data.remove(key)
    return data


print('Loading labels...')
# %%
import pickle as pkl

filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total_2.pkl'
with open(filename, 'rb') as f:
    data, energy, labels, position = pkl.load(f)

# %%
print(eventList_total[2676])
print(position[eventList_total[2676]])
# %%
data = dict()
# %%
print('Solving sponi...')
data['train'] = clean_missing_data(partition['train'], position)

print(len(data['train']))
# %%
print(len(position.keys()))
# %%
print(len(data['train']))
# %%
data['test'] = clean_missing_data(data['test'], position)
# %%
data['validation'] = clean_missing_data(data['validation'], position)
