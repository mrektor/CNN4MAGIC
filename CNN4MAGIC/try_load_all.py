import numpy as np
import pickle
from glob import glob
from tqdm import tqdm

# %
def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

# %%
proton_df = pickle_read('/data4T/magic_data/protons_big_df_ID_Labels.pkl')
# %%
proton_list = proton_df['ID'].values

# %%
from tqdm import tqdm

all_protons = []
for single_filepath in tqdm(proton_list):
    single_event = np.load(f'/data4T/magic_data/interpolated_protons/{single_filepath}.npy')
    all_protons.append(single_event)
all_protons = np.array(all_protons)

# %%
pickle_dump('/ssdraptor/magic_data/classification_MC/all_protons.pkl', all_protons)
#%%
all_protons = []
for single_filepath in tqdm(proton_list):
    single_event = np.load(f'/data4T/magic_data/interpolated_protons/{single_filepath}.npy')
    all_protons.append(single_event)
all_protons = np.array(all_protons)

#%%

point_list = glob('/data4T/quellocheerainssdraptor/magic_data/data_processed/point_like/*.npy')
print(len(point_list))

#%
all_point = []
for single_filepath in tqdm(point_list):
    single_event = np.load(single_filepath)
    all_point.append(single_event)
all_point = np.array(all_point)
#%%
pickle_dump('/ssdraptor/magic_data/classification_MC/all_point.pkl', all_point)
#%%
all_point.shape
#%%
import time
bef=time.time()
all_protons=pickle_read('/ssdraptor/magic_data/classification_MC/all_protons.pkl')
print(f'Time for loading protons: {time.time()-bef}')
#%%

