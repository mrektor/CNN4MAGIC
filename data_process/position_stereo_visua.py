import os

import matplotlib.pyplot as plt
import numpy as np
import uproot

from data_process.InterpolateMagic import InterpolateMagic


def read_from_root(filename):
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

    f = uproot.open(filename)

    tree = f['Events']
    ids = np.arange(tree.numentries)
    df = tree.pandas.df(ARRAY_COLUMNS.keys())
    df.rename(columns=ARRAY_COLUMNS, inplace=True)

    df['event_id'] = ids
    df = df[df['stereo_evt_number'] > 0]  # Select the events that triggered in both telescopes

    df2 = tree.pandas.df(PIXEL_COLUMNS.keys())  # The dataframe containing the pixel data
    df2.rename(columns=PIXEL_COLUMNS, inplace=True)

    # Clean messy values
    df2['photon_time'].loc[df2['photon_time'] < 0] = 30
    df2['photon_time'].loc[df2['photon_time'] > 60] = 30

    # EVENT START FROM 1
    event_idx = df['event_id'].values

    time = df2['photon_time'].loc[event_idx].unstack(level=-1)
    phe = df2['phe'].loc[event_idx].unstack(level=-1)

    return df, phe, time


def stereo_interp_from_root(filenames):
    filenameM1 = filenames[0]
    filenameM2 = filenames[1]

    if filenameM1[-26:-7] != filenameM2[-26:-7]:
        print('Ostia! filename are different: ', filenameM1, filenameM2)
        return None  # Escape

    if os.stat(filenameM1).st_size == 0:
        print('Empty file: ' + filenameM1)
        return None

    if os.stat(filenameM2).st_size == 0:
        print('Empty file: ' + filenameM2)
        return None

    df1, phe1, time1 = read_from_root(filenameM1)
    df2, phe2, time2 = read_from_root(filenameM2)

    interpolator = InterpolateMagic(15)
    num_events = df1.shape[0]
    m1_interp = np.zeros((num_events, 2, 67, 68))
    m2_interp = np.zeros((num_events, 2, 67, 68))
    pos_interp1 = np.zeros((num_events, 2))
    pos_interp2 = np.zeros((num_events, 2))
    for idx in range(len(phe1)):
        m1_interp[idx, 0, :, :] = interpolator.interpolate(time1.iloc[idx, :1039].values, remove_last=False, plot=False)
        m1_interp[idx, 1, :, :] = interpolator.interpolate(phe1.iloc[idx, :1039].values, remove_last=False, plot=False)
        pos_interp1[idx, :] = interpolator.interp_pos([df1['srcpos_x'].iloc[idx], df1['srcpos_y'].iloc[idx]])

        m2_interp[idx, 0, :, :] = interpolator.interpolate(time2.iloc[idx, :1039].values, remove_last=False, plot=False)
        m2_interp[idx, 1, :, :] = interpolator.interpolate(phe2.iloc[idx, :1039].values, remove_last=False, plot=False)
        pos_interp2[idx, :] = interpolator.interp_pos([df2['srcpos_x'].iloc[idx], df2['srcpos_y'].iloc[idx]])

    result = {'corsika_event_number_1': df1['corsika_event_number'].values,
              'corsika_event_number_2': df2['corsika_event_number'].values,
              'energy': df1['energy'].values,
              'src_X1': df1['srcpos_x'], 'src_Y1': df1['srcpos_y'],
              'src_X2': df2['srcpos_x'], 'src_Y2': df2['srcpos_y'],
              'pos_interp1': pos_interp1, 'pos_interp2': pos_interp2,
              'M1_interp': m1_interp, 'M2_interp': m2_interp}
    return result


# %%
file1 = '/data/mariotti_data/download_magic/MC/GA_M1_za05to35_8_821320_Y_w0.root'
file2 = '/data/mariotti_data/download_magic/MC/GA_M2_za05to35_8_821320_Y_w0.root'
res = stereo_interp_from_root([file1, file2])

# %%
fig, axs = plt.subplots(10, 4, figsize=(6 * 2, 12 * 2))
idx = 1
for row in range(10):
    axs[row, 0].imshow(res['M1_interp'][idx, 1, :, :])
    axs[row, 0].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=21)
    axs[row, 0].set_title('M1, event ' + str(idx))

    axs[row, 1].imshow(res['M1_interp'][idx, 0, :, :])
    axs[row, 1].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=21)
    axs[row, 1].set_title('M1, Time')

    axs[row, 2].imshow(res['M2_interp'][idx, 1, :, :])
    axs[row, 2].plot(res['pos_interp2'][idx][0], res['pos_interp2'][idx][1], 'xr', markersize=21)
    axs[row, 2].set_title('M2, event ' + str(idx))

    axs[row, 3].imshow(res['M2_interp'][idx, 0, :, :])
    axs[row, 3].plot(res['pos_interp2'][idx][0], res['pos_interp2'][idx][1], 'xr', markersize=21)
    axs[row, 3].set_title('M2, Time')
    idx += 1

plt.tight_layout()
# fig.suptitle('Event ' + str(idx) + ' Energy = ' + str(res['energy'][idx]))
plt.savefig('/data/mariotti_data/data_process/energy_time_pos/fig' + str(idx) + '.png')
plt.savefig('/data/mariotti_data/data_process/energy_time_pos/fig' + str(idx) + '.eps')
plt.savefig('/data/mariotti_data/data_process/energy_time_pos/fig' + str(idx) + '.pdf')

# plt.show()
