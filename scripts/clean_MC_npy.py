import glob
import os
import time

import numpy as np

fileList = glob.glob('/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/*.npy')

problems = []
bef = time.time()
for file in fileList:
    try:
        a = np.load(file)
    except:
        problems.append(file)
        os.system('mv ' + file + ' ' + '/data2T/mariotti_data_2/MC_npy/finish_dump_MC/problems')

now = time.time()
print(f'Time elapsed: {(now-bef)/60} min')
