import math

import h5py
import matplotlib.pyplot as plt
import numpy as np

pic_folder = 'notebooks/pic_folders'

# %%


###### Read files #######
filename = "notebooks/RandomE_1e6.h5"
f = h5py.File(filename, 'r')
a_group_key = list(f.keys())[0]
e = np.array(f[a_group_key])

# We take as triggered energies an array of 10% the size of the original
e_trig = e[np.random.choice(len(e), size=int(len(e) * 0.1), replace=False)]

Simulated_Events = np.size(e)
Triggered_Events = Simulated_Events * 0.1
fig, ax = plt.subplots()
ax.hist(np.log10(e))
ax.set_yscale("log")
plt.savefig(f'{pic_folder}/hist1.png')
plt.close()

# %%

##### Binnings and constants######
# Whenever implemented using simulated files, most of these values can be read from the simulations
emin = 50.  # GeV
emax = 30.e3  # GeV
eedges = 13
ebins = eedges - 1
E = np.logspace(math.log10(emin), math.log10(emax), eedges)
Emed = np.sqrt(E[:-1] * E[1:])

gammaness_bins = 10
theta2_bins = 10

# Maximum impact parameter simulated for low Zd is 350 m, therefore Area_sim is pi*350^2
Max_impact = 350.e2  # in cm
Area_sim = math.pi * math.pow(Max_impact, 2)  # cm^2

# Weighting of the MC from -2.0 to -2.6
Index_sim = -1.6
Index_Crab = -2.62

# Only applicable for the simulation of diffuse gamma or protons
cone = 0.
