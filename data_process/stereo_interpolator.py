import glob
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import uproot
from ctapipe.image import tailcuts_clean, hillas_parameters, leakage, concentration
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.instrument import CameraGeometry

from data_process.InterpolateMagic import InterpolateMagic


def compute_stuff(phe_df, time_df, only_relevant=False):
    camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')
    all_events = []
    for i in range(phe_df.shape[0]):
        event_image = phe_df.iloc[i, :1039]
        clean = tailcuts_clean(camera_MAGIC, event_image, picture_thresh=6, boundary_thresh=4)
        event_image_cleaned = event_image.copy()
        event_image_cleaned[~clean] = 0

        all_data = {}

        hillas_params = hillas_parameters(camera_MAGIC, event_image_cleaned)
        leakage_params = leakage(camera_MAGIC, event_image, clean)

        all_data.update(hillas_params)
        all_data.update(leakage_params)

        if not only_relevant:
            event_time = time_df.iloc[i, :1039]
            conc = concentration(camera_MAGIC, event_image, hillas_params)
            # n_islands, island_id = number_of_islands(camera_MAGIC, clean)
            timing = timing_parameters(
                camera_MAGIC[clean],
                event_image[clean],
                event_time[clean],
                hillas_params,
            )
            all_data.update(conc)
            all_data.update(timing)
        all_events.append(all_data)

    df2 = pd.DataFrame(all_events)
    return df2


def read_from_root(filename, want_extra=False, pruning=False):
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

    # Compute hillas parameters, leakage and other hand-crafted features
    if want_extra:
        extras = compute_stuff(phe, time, only_relevant=True)

        # Filter with some criterion
        if pruning:
            intensity_ok = extras['intensity'] > 100
            leak_ok = extras['leakage1_pixel'] < 0.2
            condition = np.logical_and(intensity_ok, leak_ok)

            return df[condition.values], extras[condition.values], phe[condition.values], time[condition.values]

        return df, extras, phe, time

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

    df1, extras1, phe1, time1 = read_from_root(filenameM1, want_extra=True, pruning=False)
    df2, extras2, phe2, time2 = read_from_root(filenameM2, want_extra=True, pruning=False)

    interpolator = InterpolateMagic(15)
    num_events = df1.shape[0]
    m1_interp = np.zeros((num_events, 67, 68, 2))
    m2_interp = np.zeros((num_events, 67, 68, 2))
    pos_interp1 = np.zeros((num_events, 2))
    pos_interp2 = np.zeros((num_events, 2))
    for idx in range(len(phe1)):
        m1_interp[idx, :, :, 0] = interpolator.interpolate(time1.iloc[idx, :1039].values, remove_last=False, plot=False)
        m1_interp[idx, :, :, 1] = interpolator.interpolate(phe1.iloc[idx, :1039].values, remove_last=False, plot=False)
        pos_interp1[idx, :] = interpolator.interp_pos([df1['srcpos_x'].iloc[idx], df1['srcpos_y'].iloc[idx]])

        m2_interp[idx, :, :, 0] = interpolator.interpolate(time2.iloc[idx, :1039].values, remove_last=False, plot=False)
        m2_interp[idx, :, :, 1] = interpolator.interpolate(phe2.iloc[idx, :1039].values, remove_last=False, plot=False)
        pos_interp2[idx, :] = interpolator.interp_pos([df2['srcpos_x'].iloc[idx], df2['srcpos_y'].iloc[idx]])

    result = {'corsika_event_number_1': df1['corsika_event_number'].values,
              'corsika_event_number_2': df2['corsika_event_number'].values,
              'energy': df1['energy'].values,
              'src_X1': df1['srcpos_x'], 'src_Y1': df1['srcpos_y'],
              'src_X2': df2['srcpos_x'], 'src_Y2': df2['srcpos_y'],
              'extras1': extras1,
              'extras2': extras2,
              'pos_interp1': pos_interp1, 'pos_interp2': pos_interp2,
              'M1_interp': m1_interp, 'M2_interp': m2_interp}

    with open('/data2T/mariotti_data_2/interp_from_root/MC_channel_last_full/result_' + filenameM1[-26:-7] + '.pkl',
              'wb') as f:
        pickle.dump(result, f, protocol=4)
    print(f'Saved {filenameM1[-26:-7]}')

    # return result


def stereo_interp_from_txt(filenames):
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

    m1 = pd.read_csv(filenameM1, sep=' ', header=None)
    m2 = pd.read_csv(filenameM2, sep=' ', header=None)

    if m1.shape[1] < 1000:
        print('OSTIA! It is not full in columns the file:' + filenameM1)
        return None

    if m2.shape[1] < 1000:
        print('OSTIA! It is not full in columns the file:' + filenameM2)
        return None

    # put int trigger1 the values of the trigger of the txt that have the highest number of simulated events.
    # Not really elegant, but it should work.
    if m1.iloc[:, 0].shape[0] < m2.iloc[:, 0].shape[0]:
        trigger1 = np.array(m1.iloc[:, 0])
        trigger2 = np.array(m2.iloc[:, 0])
    else:
        trigger2 = np.array(m1.iloc[:, 0])
        trigger1 = np.array(m2.iloc[:, 0])

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

    result = {'trigger': trigger_values,
              'energy': energy,
              'positionM1': position_1, 'positionM2': position_2,
              'M1_interp': m1_interp, 'M2_interp': m2_interp}
    print(filenameM1[-26:-7])
    with open('/data2T/mariotti_data_2/src_pos_cam/result_' + filenameM1[-26:-7] + '.pkl',
              'wb') as f:
        pickle.dump(result, f, protocol=4)


# %%
# Load all the filenames

fileM1 = glob.glob('/data/mariotti_data/download_magic/MC/GA_M1_*.root')
fileM2 = glob.glob('/data/mariotti_data/download_magic/MC/GA_M2_*.root')


def get_pair_match(a, b):
    result = []
    for i in a:
        for j in b:
            if i[-26:-7] == j[-26:-7]:
                result.append((i, j))
    return result


mFull = get_pair_match(fileM1, fileM2)

# %%
# Start the parallel computing
print('start multiprocessing')
pool = multiprocessing.Pool(processes=18)
pool.map(stereo_interp_from_root, mFull)
pool.close()
pool.join()
print('All done, everything is fine')
