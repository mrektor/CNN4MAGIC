import glob

from tqdm import tqdm

# %%
filelist = glob.glob('/data2T/mariotti_data_2/MC_npy/complementary_computation/e*')

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
with open('/data2T/mariotti_data_2/MC_npy/complementary_dump_total.pkl', 'wb') as f:
    pickle.dump((partition, energy_total, labels_total, position_total), f)

#############
# %%
to_move = glob.glob('/data2T/mariotti_data_2/MC_npy/partial_dump_MC/*.npy')
print(len(to_move))
# %%
print(to_move[0])
# %%
import os

for idx in tqdm(range(1, len(to_move))):
    os.system('mv ' + to_move[idx] + ' /data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/')
