import glob
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd

from InterpolateMagic import InterpolateMagic


def stereo_interp_from_txt(filenames):
    filenameM1 = filenames[0]
    filenameM2 = filenames[1]

    if filenameM1[-26:-7] != filenameM2[-26:-7]:
        print('Ostia! filename are different: ', filenameM1, filenameM2)
        return None  # Escape

    if os.stat(filenameM1).st_size != 0:
        print('Empty file: ' + filenameM1)
        return None

    if os.stat(filenameM2).st_size != 0:
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

    m1_idx_mask = np.isin(trigger1, trigger2)  # TODO: check the smallest then do the isin accordingly

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
    with open('/data2T/mariotti_data_2/energy_MC_diffuse/result_' + filenameM1[-26:-7] + '.pkl',
              'wb') as f:
        pickle.dump(result, f, protocol=4)


# %%
# Load all the filenames
fileM1 = glob.glob('/data/mariotti_data/CNN4MAGIC/dataset/MC/Diffuse/M1/GA_M1_*.txt')
fileM2 = glob.glob('/data/mariotti_data/CNN4MAGIC/dataset/MC/Diffuse/M2/GA_M2_*.txt')


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
pool = multiprocessing.Pool(processes=16)
pool.map(stereo_interp_from_txt, mFull)
pool.close()
pool.join()
print('All done, everything is fine')
