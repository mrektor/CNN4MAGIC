import numpy as np
import pandas as pd

# %%
m1 = pd.read_csv('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M1/GA_M1_za05to35_8_821318_Y_w0.txt',
                 sep=" ",
                 header=None)
m2 = pd.read_csv('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M2/GA_M2_za05to35_8_821318_Y_w0.txt',
                 sep=' ', header=None)

# %%
trigger1 = np.array(m1.iloc[:, 0])
trigger2 = np.array(m2.iloc[:, 0])

m1_idx_mask = np.isin(trigger1, trigger2)
m2_idx_mask = np.isin(trigger2, trigger1)

m1_both_event = m1[m1_idx_mask]
m2_both_event = m2[m2_idx_mask]

# %%
