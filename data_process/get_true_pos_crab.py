import pickle
from glob import glob

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm


def is_zenith_ok(filename, zenith_upper_limit=35):
    # print('opening root file')
    f = uproot.open(filename)

    # Zenith check:
    drive_tree = f['Drive']
    df_zenith = drive_tree.pandas.df({'MReportDrive.fCurrentZd'})
    global_zenith = np.mean(df_zenith.values)
    if global_zenith < zenith_upper_limit:
        # print(f'file {filename} has a Zenith of {global_zenith}, thus ok.')
        return True
    else:
        # print(f'file {filename} has a Zenith of {global_zenith}, thus not considering it.')
        return False


def read_from_root_realdata_position_crab(filename):
    ARRAY_COLUMNS = {
        'MRawEvtHeader.fStereoEvtNumber': 'stereo_evt_number',
        'MTime.fTime.fMilliSec': 'millisec',
        'MTime.fNanoSec': 'nanosec',
        'MRawEvtHeader.fTimeDiff': 'timediff',
        'MSrcPosCam.fX': 'xcoord',
        'MSrcPosCam.fY': 'ycoord',

    }

    f = uproot.open(filename)

    tree = f['Events']
    ids = np.arange(tree.numentries)
    df = tree.pandas.df(ARRAY_COLUMNS.keys())
    df.rename(columns=ARRAY_COLUMNS, inplace=True)

    df['event_id'] = ids
    df = df[df['stereo_evt_number'] > 0]  # Select the events that triggered in both telescopes

    return df


def load(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res


crab_complement_list = glob('/data/magic_data/crab_complement_2/*.pkl')

big_df_complement_crabpos = pd.DataFrame()
full_event_list = []
for single_complement in tqdm(crab_complement_list):
    identifier = single_complement[52:67]
    file_for_pos = glob(f'/data/magic_data/crab_for_true_pos/*{identifier}*')[0]
    df_pos_tmp = read_from_root_realdata_position_crab(file_for_pos)

    df_complement, evt_list = load(single_complement)[1:3]
    num_events_complement = df_complement.shape[0]

    df_complement['xcoord_crab'] = df_pos_tmp['xcoord']
    df_complement['ycoord_crab'] = df_pos_tmp['ycoord']
    df_complement['xcoord_crab'] = df_complement['xcoord_crab'].interpolate()
    df_complement['ycoord_crab'] = df_complement['ycoord_crab'].interpolate()

    full_event_list = full_event_list + evt_list
    big_df_complement_crabpos = big_df_complement_crabpos.append(df_complement)

# %%
with open('/data/magic_data/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl', 'wb') as f:
    pickle.dump((big_df_complement_crabpos, full_event_list), f)
