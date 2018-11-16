# import logging

import numpy as np
import uproot

# %%

ARRAY_COLUMNS = {
    'MMcEvt.fEvtNumber': 'corsika_event_number',
    'MMcEvt.fEnergy': 'energy',
    'MMcEvt.fTheta': 'theta',
    'MMcEvt.fPhi': 'phi',
    'MMcEvt.fCoreX': 'core_x',
    'MMcEvt.fCoreY': 'core_y',
    'MMcEvt.fImpact': 'impact',
    'MMcEvt.fTelescopePhi': 'telescope_phi',
    'MMcEvt.fTelescopeTheta': 'telescope_theta',
    'MSrcPosCam.fX': 'srcpos_x',
    'MSrcPosCam.fY': 'srcpos_y',
    'MRawEvtHeader.fStereoEvtNumber': 'stereo_evt_number'
}

PIXEL_COLUMNS = {
    'MArrivalTime.fData': 'photon_time',
    'MCerPhotEvt.fPixels.fPhot': 'phe',
}

# logging.info('Opening file')result_a05to35_8_821325_Y_
filename = '/data/mariotti_data/download_magic/MC/GA_M2_za05to35_8_821325_Y_w0.root'
f = uproot.open(filename)

# logging.info('Getting tree')
tree = f['Events']
# branches = set(k.decode('ascii') for k in tree.keys())
ids = np.arange(tree.numentries)

df = tree.pandas.df(ARRAY_COLUMNS.keys())
df.rename(columns=ARRAY_COLUMNS, inplace=True)
df['event_id'] = ids

df = df[df['stereo_evt_number'] > 0]  # Select the non-Zero elements

# %%
df2 = tree.pandas.df(PIXEL_COLUMNS.keys())
df2.rename(columns=PIXEL_COLUMNS, inplace=True)
# %%

df2['photon_time'].loc[df2['photon_time'] < 0] = 30
df2['photon_time'].loc[df2['photon_time'] > 60] = 30

# %% EVENT START FROM 1
event_idx = df['event_id'].values
# %%
event_idx = 34
# event_idx=event_idx+1
time = df2['photon_time'][event_idx].unstack(level=-1)
phe = df2['phe'][event_idx].unstack(level=-1)
print(df['energy'].iloc[event_idx])
# %%
# from InterpolateMagic import InterpolateMagic
from data_process.InterpolateMagic import InterpolateMagic

# %%
interpolator = InterpolateMagic(15)
en = [df['srcpos_x'].iloc[event_idx], df['srcpos_y'].iloc[event_idx]]
# %%
loc = interpolator.interpolate(en, remove_last=False, plot=False)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(time.values)
plt.plot(phe.values)
plt.legend(['time', 'phe'])
plt.title('Event ' + str(event_idx) + '. Energy = ' + str(df['energy'].iloc[event_idx]))
# plt.show()
plt.savefig('time_phe_fig' + str(event_idx) + '_M2.png')


# %% Sanity Check
print(phe.shape)
print(time.shape)
print(event_idx)

# pix = df2['phe'][event_idx]
