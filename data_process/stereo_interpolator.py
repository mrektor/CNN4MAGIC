import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext

# %%
config = SparkConf().setAppName('Homework 2').setMaster('local[*]')
sc = SparkContext(conf=config)

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
# m2_idx_mask = np.isin(trigger2, trigger1)

m1_both_event = m1[m1_idx_mask, :]
# m2_both_event = m2[m2_idx_mask, :] # m1 and m2 are equal


# %%
trigger_values = m1.iloc[m1_idx_mask, 0].values
position_1 = m1.iloc[m1_idx_mask, 2:4].values
position_2 = m2.iloc[m1_idx_mask, 2:4].values
energy = m1.iloc[m1_idx_mask, 1].values
pixel_m1 = m1.iloc[m1_idx_mask, 2 + 2:1041 + 2].values
pixel_m2 = m2.iloc[m1_idx_mask, 2 + 2:1041 + 2].values

# %%
import matplotlib.pyplot as plt
from InterpolateMagic import InterpolateMagic

# import data_process.InterpolateMagic

# %%
interpolator = InterpolateMagic(15)
idx = 3
a = interpolator.interpolate(pixel_m1[idx], remove_last=False, plot=False)  # Test
b = interpolator.interpolate(pixel_m2[idx], remove_last=False, plot=False)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(a)
plt.title('M1')
# plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(b)
# plt.colorbar()
plt.title('M2')
plt.tight_layout()
plt.savefig('test3.png')
