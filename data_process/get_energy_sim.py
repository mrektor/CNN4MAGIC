import pandas as pd
import uproot


def get_energy_sim(root_filename):
    ARRAY_COLUMNS = {
        # 'MMcEvt.fEnergy': 'energy',
        'MMcEvtBasic.fEnergy': 'energy_org',
    }

    f = uproot.open(root_filename)

    tree = f['OriginalMC']
    # branches = set(k.decode('ascii') for k in tree.keys())
    # dfs = []
    df = tree.pandas.df(ARRAY_COLUMNS.keys())
    df.rename(columns=ARRAY_COLUMNS, inplace=True)
    return df


# %%
from glob import glob

list = glob('/home/emariott/deepmagic/data_root/mc/point_like/*M2*.root')
print(len(list))
# %%
print(list[:2])
# %%
from tqdm import tqdm

full_df = pd.DataFrame()
for file in tqdm(list):
    tmp = get_energy_sim(file)
    full_df = full_df.append(tmp)

print(full_df.shape)
# %%
print(full_df.keys())

energy_sim = full_df['energy_org'].values

# %%
import pickle

with open('/home/emariott/deepmagic/output_data/for_sensitivity/energy_sim_test_point.pkl', 'wb') as f:
    pickle.dump(energy_sim, f)

# %%
energy_sim.shape
