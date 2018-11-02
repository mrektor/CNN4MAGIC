import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext

# %%
config = SparkConf().setAppName('Homework 2').setMaster('local[*]')
sc = SparkContext(conf=config)
import matplotlib.pyplot as plt
from InterpolateMagic import InterpolateMagic

# %%
m1 = pd.read_csv('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M1/GA_M1_za05to35_8_821318_Y_w0.txt',
                 sep=" ",
                 header=None)
m2 = pd.read_csv('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M2/GA_M2_za05to35_8_821318_Y_w0.txt',
                 sep=' ', header=None)


trigger1 = np.array(m1.iloc[:, 0])
trigger2 = np.array(m2.iloc[:, 0])

m1_idx_mask = np.isin(trigger1, trigger2)

trigger_values = m1.iloc[m1_idx_mask, 0].values
position_1 = m1.iloc[m1_idx_mask, 2:4].values
position_2 = m2.iloc[m1_idx_mask, 2:4].values
energy = m1.iloc[m1_idx_mask, 1].values
pixel_m1 = m1.iloc[m1_idx_mask, 2 + 2:1041 + 2].values
pixel_m2 = m2.iloc[m1_idx_mask, 2 + 2:1041 + 2].values

interpolator = InterpolateMagic(15)
m1_interp = [interpolator.interpolate(instance, remove_last=False, plot=False) for instance in pixel_m1]
m2_interp = [interpolator.interpolate(instance, remove_last=False, plot=False) for instance in pixel_m2]

# %% Test
print(len(m1_interp))
# %%

idx = 3

#Plot
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(m1_interp[idx])
plt.title('M1')

plt.subplot(1, 2, 2)
plt.imshow(m2_interp[idx])
plt.title('M2')
plt.tight_layout()
plt.savefig('test.png')
