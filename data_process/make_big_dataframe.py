import glob
import pickle

import pandas as pd
from tqdm import tqdm

# %%
filelist = glob.glob('/data4T/magic_data/complement_protons/*.pkl')

# %%
print(len(filelist))
with open(filelist[0], 'rb') as f:
    event_idx_list, labels, df1, df2, extras1, extras2 = pickle.load(f)
# %%
print(len(event_idx_list), len(labels), extras1.shape)
#%%
big_df = pd.DataFrame()

for file in tqdm(filelist):
    with open(file, 'rb') as f:
        eventList, labels, df1, df2, extras1, extras2 = pickle.load(f)

    df_next = pd.DataFrame({'ID': eventList,
                            # 'label': labels,
                            # 'energy': df1['energy'].values,
                            # 'srcpos_x': df1['srcpos_x'].values,
                            # 'srcpos_y': df1['srcpos_y'].values,
                            # 'theta': df1['theta'].values,
                            # 'core_x': df1['core_x'].values,
                            # 'core_y': df1['core_y'].values,
                            # 'impact_M1': df1['impact'].values,
                            # 'impact_M2': df2['impact'].values,
                            # 'intensity_M1': extras1['intensity'].values,
                            # 'intensity_M2': extras2['intensity'].values,
                            # 'leakage2_pixel_M1': extras1['leakage2_pixel'].values,
                            # 'leakage2_pixel_M2': extras2['leakage2_pixel'].values
                            })
    big_df = big_df.append(df_next)
# %%
print(big_df.shape)
#%%
big_df['label'] = 0
#%%
print(big_df.shape)
# %%
gold = big_df['impact'] < 8000
golden_df = big_df[gold]

# %
print(golden_df.shape)
# %%

with open('/data4T/magic_data/complement_protons/protons_big_df_ID_Labels.pkl', 'wb') as f:
    pickle.dump(big_df, f)
#%%
with open('/ssdraptor/magic_data/classification_MC/protons_big_df_ID_Labels.pkl', 'wb') as f:
    pickle.dump(big_df, f)