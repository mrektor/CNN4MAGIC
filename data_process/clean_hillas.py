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

phe = phe1.iloc[idx, :1039]
print(phe.shape)
time = time1.iloc[idx, :1039]

# %
camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')

# %%

from ctapipe.image import tailcuts_clean

# from ctapipe.image.cleaning import number_of_islands

event_image = phe
# %%
print(event_image.shape)
# %%
clean = tailcuts_clean(camera_MAGIC, event_image, picture_thresh=6, boundary_thresh=4)
event_image_cleaned = event_image.copy()
event_image_cleaned[~clean] = 0
print('ok')
# %%
from ctapipe.image import hillas_parameters, leakage, concentration
from ctapipe.image.timing_parameters import timing_parameters

hillas_params = hillas_parameters(camera_MAGIC, event_image_cleaned)

print(hillas_params)

# %%
from ctapipe.image.cleaning import number_of_islands

l = leakage(camera_MAGIC, phe, clean)
conc = concentration(camera_MAGIC, phe, hillas_params)
n_islands, island_id = number_of_islands(camera_MAGIC, clean)
timing = timing_parameters(
    camera_MAGIC[clean],
    phe[clean],
    time[clean],
    hillas_params,
)

print(timing)
print(n_islands)
print(conc)
print(l)

# %%
from ctapipe.image import tailcuts_clean, hillas_parameters, leakage, concentration
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.cleaning import number_of_islands


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
            n_islands, island_id = number_of_islands(camera_MAGIC, clean)
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


# %%
print(phe1.shape)
# %%
testissimo2 = compute_stuff(phe1, time1, only_relevant=False)

# %%
a = []
b = {'c': 2, 'd': 3}
c = {'c': 10, 'd': 15}

a.append(b)
a.append(c)
# %%
import pandas as pd

df_test = pd.DataFrame(a)
