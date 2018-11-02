import glob
import multiprocessing
import pickle

import numpy as np
import pandas as pd

from InterpolateMagic import InterpolateMagic


def stereo_interp_from_txt(filenames):
    filenameM1 = filenames[0]
    filenameM2 = filenames[1]

    if filenameM1[-26:-7] != filenameM2[-26:-7]:
        print('Ostia! filename are different: ', filenameM1, filenameM2)
        raise ValueError

    m1 = pd.read_csv(filenameM1, sep=' ', header=None)
    m2 = pd.read_csv(filenameM2, sep=' ', header=None)

    trigger1 = np.array(m1.iloc[:, 0])
    trigger2 = np.array(m2.iloc[:, 0])

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
    with open('/data/mariotti_data/CNN4MAGIC/pickle_data/processed/result_' + filenameM1[-26:-7] + '.pkl',
              'wb') as f:
        pickle.dump(result, f, protocol=4)


# Load all the filenames
file1 = glob.glob('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M1/GA_M1_*.txt')
file2 = glob.glob('/data/mariotti_data/CNN4MAGIC/dataset/MC/Energy_SrcPosCam/M2/GA_M2_*.txt')

list_of_files = np.array([file1, file2]).T.tolist()  # Each line here is a couple of filenames of the same simulation

# Start the parallel computing
print('start multiprocessing')
pool = multiprocessing.Pool(processes=16)
pool.map(stereo_interp_from_txt, list_of_files)
pool.close()
pool.join()
print('All done, everything is fine')
