import glob
import pickle

from tqdm import tqdm

# %
filelist = glob.glob('/data/magic_data/clean_10_5/crab/complement/*.pkl')

# %
print(len(filelist))

# %%
with open(filelist[0], 'rb') as f:
    eventList, labels = pickle.load(f)
    # eventList, labels, energy, position, df1, df2 = pickle.load(f)


# %%
import pickle
import pandas as pd
eventList_total = []
energy_total = {}
labels_total = {}
position_total = {}
df_big = pd.DataFrame()
extra_big = pd.DataFrame

for file in tqdm(filelist):
    with open(file, 'rb') as f:
        eventList, labels = pickle.load(f)
        # eventList, labels, energy, position, df1, df2 = pickle.load(f)
        # eventList, labels, energy, position, df1, df2, extra1, extra2 = pickle.load(f)
    eventList_total = eventList_total + eventList
    # energy_total.update(energy)
    labels_total.update(labels)
    # position_total.update(position)
    # df_big = df_big.append(df1)
    # extra_big = extra_big.append(extra1)

# %%
print(len(eventList_total))
print(len(labels_total.keys()))
# %%
print(len(position_total))
# %%
with open('/data/magic_data/clean_10_5/crab/events_labels_crab_clean10_5.pkl', 'wb') as f:
    pickle.dump((eventList_total, labels_total), f)


# %%
with open('/data/magic_data/clean_10_5/point_MC/point_clean_10_5_complement.pkl', 'wb') as f:
    pickle.dump((eventList_total, labels_total, energy_total, position_total), f)

# %%
with open('/data/magic_data/clean_10_5/point_MC/df_big_10_5.pkl', 'wb') as f:
    pickle.dump(df_big, f)

# %%
filename = '/data/magic_data/MC_npy/complementary_dump_total_2.pkl'
with open(filename, 'rb') as f:
    _, energy, labels_MC, position = pickle.load(f)

# %%
mc_root_labels = {}
mc_root_labels.update(labels_MC)
mc_root_labels.update(labels_total)
print(len(mc_root_labels))
# %%
with open('/data/magic_data/mc_root_labels.pkl', 'wb') as f:
    pickle.dump(mc_root_labels, f)

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
# %%

from CNN4MAGIC.Generator.gen_util import load_data_generators

tr, va, ene = load_data_generators(1, want_energy=True)

# %%
tr[0]
# %%
import time

i = 0
bb = 0
bef = time.time()
for a, b in enumerate(tr):
    bb = b
    i += 1
now = time.time()
print(i)
print(now - bef)
# %%
i = 0
bef = time.time()
problems = []
for a in range(57430, len(tr)):
    try:
        bb = tr[a]
    except:
        problems.append(a)
    i += 1
now = time.time()
print(i)
print(now - bef)

# %%
bef = time.time()
for a, b in enumerate(va):
    bb = b
    i += 1
now = time.time()
print(i)
print(now - bef)
