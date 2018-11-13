import logging

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

logging.info('Opening file')
filename = '/data/mariotti_data/download_magic/MC/GA_M2_za05to35_8_824312_Y_w0.root'
f = uproot.open(filename)
# f = uproot.open('../GA_za05to35_8_S_w0_1.root')

logging.info('Getting tree')
tree = f['Events']
branches = set(k.decode('ascii') for k in tree.keys())
ids = np.arange(tree.numentries)
dfs = []

df = tree.pandas.df(ARRAY_COLUMNS.keys())
df.rename(columns=ARRAY_COLUMNS, inplace=True)

df = df[df['stereo_evt_number'] > 0]

# %%
df2 = tree.pandas.df(PIXEL_COLUMNS.keys())
# df2 = df2[df['stereo_evt_number']>0]
df2.rename(columns=PIXEL_COLUMNS, inplace=True)
# %% EVENT START FROM 1
event_idx = df['event_id'].values
# %%
time = df2['photon_time'].loc[event_idx].unstack(level=-1)
phe = df2['phe'].loc[event_idx].unstack(level=-1)
# %%
print(phe.shape)
print(time.shape)
print(event_idx.shape)

# pix = df2['phe'][event_idx]
