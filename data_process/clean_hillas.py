import ctapipe
import numpy as np
import uproot

print(ctapipe.__file__)
print(ctapipe.__version__)


# %%
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


# %%

filenameM1 = '/data/mariotti_data/download_magic/MC/GA_M1_za05to35_8_821318_Y_w0.root'
df1, phe1, time1 = read_from_root(filenameM1)

# %%

from ctapipe.instrument import CameraGeometry

idx = 10

phe = phe1[idx][:1039]
time = phe1[idx][:1039]

# %%
camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')

# %%

from ctapipe.image import tailcuts_clean

# from ctapipe.image.cleaning import number_of_islands

event_image = phe1
clean = tailcuts_clean(camera_MAGIC, event_image, picture_thresh=6, boundary_thresh=4)
event_image_cleaned = event_image.copy()
event_image_cleaned[~clean] = 0
print('ok')
