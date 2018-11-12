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
}

PIXEL_COLUMNS = {
    'MArrivalTime.fData': 'photon_time',
    'MCerPhotEvt.fPixels.fPhot': 'phe',
}

logging.info('Opening file')
filename = '/data/mariotti_data/data_process/GA_M1_za05to35_8_855599_Y_w0.root'
f = uproot.open(filename)
# f = uproot.open('../GA_za05to35_8_S_w0_1.root')

logging.info('Getting tree')
tree = f['Events']
branches = set(k.decode('ascii') for k in tree.keys())
ids = np.arange(tree.numentries)
dfs = []

df = tree.pandas.df(ARRAY_COLUMNS.keys())
df.rename(columns=ARRAY_COLUMNS, inplace=True)
df['event_id'] = ids
df

# %%
df2 = tree.pandas.df(PIXEL_COLUMNS.keys())
df2.rename(columns=PIXEL_COLUMNS, inplace=True)
# %% EVENT START FROM 1
event_idx = 1
df2['photon_time'][event_idx]
