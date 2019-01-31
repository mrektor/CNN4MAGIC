import glob
import logging

import numpy as np
import pandas as pd
import uproot


# import click

# %%
def get_energies(filename):
    ARRAY_COLUMNS = {
        # 'MMcEvt.fEnergy': 'energy',
        'MMcEvtBasic.fEnergy': 'energy_org',
    }

    logging.info('Opening file')
    f = uproot.open(filename)

    logging.info('Getting tree')
    tree = f['OriginalMC']
    branches = set(k.decode('ascii') for k in tree.keys())
    ids = np.arange(tree.numentries)
    df = tree.pandas.df(ARRAY_COLUMNS.keys())
    df.rename(columns=ARRAY_COLUMNS, inplace=True)
    df['event_id'] = ids
    return df


# %%
files = glob.glob('/home/emariott/deepmagic/data_root/mc/point_like/*M1*.root')
# %% test
b = get_energies(files[0])
# %% do for every file
from tqdm import tqdm

big_df = pd.DataFrame()

for file in tqdm(files):
    a = get_energies(file)
    big_df = big_df.append(a)

print(big_df.shape)
# %%
import pickle

file_save = '/home/emariott/deepmagic/data_root/mc/all_energies_point.pkl'
with open(file_save, 'wb') as f:
    pickle.dump(big_df, f)
