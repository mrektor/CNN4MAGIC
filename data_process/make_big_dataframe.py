import glob
import pickle

import pandas as pd
from tqdm import tqdm

# %%
filelist = glob.glob('/home/emariott/deepmagic/data_interpolated/complementary_computation_diffuse/*.pkl')

# %%
print(len(filelist))

# %%
big_df = pd.DataFrame()

for file in tqdm(filelist):
    with open(file, 'rb') as f:
        eventList, labels, energy, position, df1, df2 = pickle.load(f)

    df_next = pd.DataFrame({'ID': eventList,
                            'energy': df1['energy'].values,
                            'srcpos_x': df1['srcpos_x'].values,
                            'srcpos_y': df1['srcpos_y'].values,
                            'impact': df1['impact'].values
                            })
    big_df = big_df.append(df_next)
# %%
print(big_df.shape)
# %%
gold = big_df['impact'] < 8000
golden_df = big_df[gold]

# %
print(golden_df.shape)
# %%

with open('/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_df.pkl', 'wb') as f:
    pickle.dump(big_df, f)
