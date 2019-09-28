import pickle
import numpy as np
from tqdm import tqdm
def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%
theta_sq_on = pickle_read('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_on.pkl')
theta_sq_off_3 = pickle_read('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_3_off.pkl')
theta_sq_off_global = pickle_read('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_off_global.pkl')
# note: theta_sq_off_global = (theta_sq_off1 + theta_sq_off2 + theta_sq_off3) / 3

energy_estimated = pickle_read('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/crab_energy_transfer-SE-inc-v3-snap-lowLR_SWA.pkl')

gammaness = pickle_read('/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/computed_data/crab_separation_efficientNet_B3_last3_lin.pkl')
#%%
chi_separation = -np.log10(1-gammaness)
#%%
theta_sq_on[:10]
#%%
print(gammaness.shape)
#%%
def significance(energy_threshold, theta_sq_threshold, gammaness_threshold, energy_estimated, theta_sq_on, theta_sq_off_global, gammaness):
    on_events = np.sum((energy_estimated.flatten() > energy_threshold) & (theta_sq_on.flatten() < theta_sq_threshold) & (gammaness.flatten() > gammaness_threshold))
    off_events = np.sum((energy_estimated.flatten() > energy_threshold) & (theta_sq_off_global.flatten() < theta_sq_threshold) & (gammaness.flatten()> gammaness_threshold))

    return (on_events-off_events)/np.sqrt(off_events)

#%%
def maximize_gridsearch(energy_estimated, theta_sq_on, theta_sq_off_global, gammaness, num_grid_steps=10):
    en_range = np.linspace(np.min(energy_estimated), np.max(energy_estimated), num_grid_steps)
    th_range = np.linspace(np.min(theta_sq_on), 0.5, num_grid_steps)
    chi_range = np.linspace(0.1, 10, num_grid_steps)
    gammaness_range = 1-10**-chi_range

    s_search = np.zeros((num_grid_steps, num_grid_steps, num_grid_steps))

    for i, e_th in enumerate(tqdm(en_range)):
        for j, t_th in enumerate(th_range):
            for k, g_th in enumerate(gammaness_range):
                s_search[i,j,k] = significance(e_th, t_th, g_th, energy_estimated, theta_sq_on, theta_sq_off_global, gammaness)

    max_significance = np.max(s_search[~np.isnan(s_search) & ~np.isinf(s_search)])
    idxs = np.where(s_search==max_significance)
    print(f'Energy th: {en_range[idxs[0]]}\nTheta2 cut: {th_range[idxs[1]]}\nGammaness cut: {gammaness_range[idxs[2]]}')
    return max_significance, en_range[idxs[0]], th_range[idxs[1]], gammaness_range[idxs[2]]
#%%
s_train, e_cut, th_cut, gamma_cut = maximize_gridsearch(energy_estimated, theta_sq_on, theta_sq_off_global, gammaness))