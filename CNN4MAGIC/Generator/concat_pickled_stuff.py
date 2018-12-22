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

# %%
len(eventList_total)
len(energy_total.keys())
# %%
num_files = len(eventList_total)
partition = {}
partition['train'] = eventList_total[:int(num_files / 2)]
partition['validation'] = eventList_total[int(num_files / 2):int(num_files * 3 / 2)]
partition['test'] = eventList_total[int(num_files * 3 / 2):]
# %%
with open('/data2T/mariotti_data_2/MC_npy/complementary_dump_total.pkl', 'wb') as f:
    pickle.dump((partition, energy_total, labels_total, position_total), f)
