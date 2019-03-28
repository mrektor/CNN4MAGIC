import pickle

import matplotlib.pyplot as plt
import numpy as np

net_name_pos = 'SEDenseNet121_position_noclean_Gold_SNAP10_minimumValidation'
dump_name = f'/home/emariott/software_magic/output_data/reconstructions/position_prediction_crab_{net_name_pos}.pkl'
with open(dump_name, 'rb') as f:
    pos_crab = pickle.load(f)

net_name_sep = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/crab_separation_{net_name_sep}.pkl'
with open(dump_name, 'rb') as f:
    separation_gammaness = pickle.load(f)

with open('/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl',
          'rb') as f:
    big_df, evt_list = pickle.load(f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        y_pred = pickle.load(f)
    print(f'loaded {filename}')
    return y_pred


energy_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/crab_energy_transfer-SE-inc-v3-snap-lowLR_SWA.pkl')

# %%
enegy_events_lin = 10 ** energy_hadrons

# %%

pos_pred = pos_crab
true_coords_mm = np.transpose(np.array([big_df['xcoord_crab'].values, big_df['ycoord_crab'].values]))
pos_true = true_coords_mm

pos_true = pos_true[:pos_pred.shape[0]]
pos_in_mm = True
if pos_in_mm:
    pos_true_deg = pos_true * 0.00337  # in deg
    pos_pred_deg = pos_pred * 0.00337  # in deg

num_events = pos_pred.shape[0]
theta_sq = np.sum((pos_true_deg - pos_pred_deg) ** 2, axis=1)
theta_sq_off_1 = np.sum((-pos_true_deg - pos_pred_deg) ** 2, axis=1)
# %%
theta_sq_off_part1 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)
theta_sq_off_part2 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] - pos_pred_deg[:, 1]) ** 2)
theta_sq_off_part3 = ((pos_true_deg[:, 0] - pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)
# theta_sq_off_mean = np.mean([theta_sq_off_part1, theta_sq_off_part3], axis=0)
fig_folder = '/home/emariott/software_magic/output_data/pictures/crab_significance_2'
plt.figure()
plt.hist(theta_sq_off_part1, log=True, bins=200)
plt.xlim([0, 0.5])
plt.savefig(f'{fig_folder}/theta_sq_off_1.png')
plt.close()
# %%

theta_sq_off1 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)
theta_sq_off2 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] - pos_pred_deg[:, 1]) ** 2)
theta_sq_off3 = ((pos_true_deg[:, 0] - pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)

theta_sq_limato = theta_sq[:len(separation_gammaness)]
theta_sq_off_1_limato = theta_sq_off1[:len(separation_gammaness)]
theta_sq_off_2_limato = theta_sq_off2[:len(separation_gammaness)]
theta_sq_off_3_limato = theta_sq_off3[:len(separation_gammaness)]

# %%
chi_gammaness = -np.log10(1 - separation_gammaness + 1e-20).flatten()
# %%

# fig_folder = '/home/emariott/software_magic/output_data/pictures/crab_significance'
plt.figure()
plt.hist(chi_gammaness, log=True, bins=200)
plt.savefig(f'{fig_folder}/chi_gammaness.png')
plt.close()

# %%
chi_steps = 10
th2_steps = 10
en_steps = 5

chi_range = np.linspace(1, 3, chi_steps)
th2_range = np.linspace(0.005, 0.05, th2_steps)
en_range = np.logspace(np.log10(220), np.log10(7133), en_steps)

signal_vect = np.zeros((chi_steps, th2_steps))
background_vect = np.zeros((chi_steps, th2_steps))

significance_vect = np.zeros((chi_steps, th2_steps))
error_vect = np.zeros((chi_steps, th2_steps))

energy_threshold = 150
print(f'Number of events rejected for their energy: {np.sum(enegy_events_lin<energy_threshold)}')
# energy_cut = enegy_events_lin > energy_threshold
for i, chi_cut in enumerate(chi_range):
    for j, th2_cut in enumerate(th2_range):
        # for k, en_cut in enumerate(en_range):
        # condition_cut = np.logical_and(chi_gammaness < chi_cut, theta_sq_limato < th2_cut)

        # signal = np.sum(theta_sq_limato[condition_cut])
        # background = np.sum(theta_sq_off_1[condition_cut])
        # signal = len(theta_sq_limato[condition_cut])
        # background = len()
        energy_cut = enegy_events_lin.flatten() > energy_threshold

        condition_cut_signal = np.logical_and(chi_gammaness > chi_cut, theta_sq_limato < th2_cut)
        condition_cut_bg_1 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_1_limato < th2_cut)
        condition_cut_bg_2 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_2_limato < th2_cut)
        condition_cut_bg_3 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_3_limato < th2_cut)

        # CUT IN ENERGY
        condition_cut_signal = np.logical_and(condition_cut_signal, energy_cut)
        condition_cut_bg_1 = np.logical_and(condition_cut_bg_1, energy_cut)
        condition_cut_bg_2 = np.logical_and(condition_cut_bg_2, energy_cut)
        condition_cut_bg_3 = np.logical_and(condition_cut_bg_3, energy_cut)

        #Compute Signal
        signal = np.sum(condition_cut_signal)
        background_1 = np.sum(condition_cut_bg_1)
        background_2 = np.sum(condition_cut_bg_2)
        background_3 = np.sum(condition_cut_bg_3)

        # Average background value
        # background = background_3
        background = (background_1 + background_2 + background_3) / 3
        error = np.sqrt((signal / np.sqrt(background)) + background * (
                    (signal - background) / (2 * (background ** (3 / 2)))) + 1 / np.sqrt(background))

        significance = (signal - background) / np.sqrt(background)
        if signal - background > 10 & (signal - background > 0.05 * background):
            significance_vect[i, j] = significance
            error_vect[i, j] = error
            background_vect[i, j] = background
            signal_vect[i, j] = signal
            # significance_vect[i, j, k] = significance
        # if significance > 35:
        # print(f'signal: {signal}, background: {background}')

significance_vect_nonan = np.nan_to_num(significance_vect)
# chi_opt_idx, th2_opt_idx, en_opt_idx = np.unravel_index(significance_vect_nonan.argmax(), significance_vect.shape)
chi_opt_idx, th2_opt_idx = np.unravel_index(significance_vect_nonan.argmax(), significance_vect.shape)
chi_opt = chi_range[chi_opt_idx]
th2_opt = th2_range[th2_opt_idx]
# en_opt = en_range[en_opt_idx]

print(f'Optimum cut in Chi: {chi_opt}')
print(f'Optimum cut in Th2: {th2_opt}')
print(
    f'Significance with these values: {significance_vect[chi_opt_idx, th2_opt_idx]} ± {error_vect[chi_opt_idx, th2_opt_idx]:.2f} with N_on={signal_vect[chi_opt_idx, th2_opt_idx]} and N_off_avg={background_vect[chi_opt_idx, th2_opt_idx]}')
# %
fig, ax = plt.subplots()
plt.imshow(significance_vect)
plt.plot(th2_opt_idx, chi_opt_idx, 'xr', markersize=20)
# plt.legend(['Max'])
plt.xlabel('$\Theta^2$ Cut')
plt.ylabel('$\chi$ Cut')
plt.xticks(range(th2_steps), [f'{value:.3f}' for value in th2_range], rotation=20)
plt.yticks(range(chi_steps), [f'{value:.1f}' for value in chi_range])
plt.title('Significance on Crab')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{fig_folder}/significance_grid_format_chi.png')
plt.savefig(f'{fig_folder}/significance_grid_format_chi.pdf')
plt.close()

# %

sensitivity = 5 / (significance_vect * np.sqrt(50 * 3600 / 1744)) * 100
# error_sensitivity = significance_vect*np.sqrt((1/background_vect)+(1/significance_vect))
chi_opt_idx, th2_opt_idx = np.unravel_index(sensitivity.argmin(), sensitivity.shape)
error_sensitivity = error_vect / significance_vect * sensitivity
fig, ax = plt.subplots()
plt.imshow(sensitivity, cmap='viridis')
plt.plot(th2_opt_idx, chi_opt_idx, 'sr', markersize=24, fillstyle='none')
# plt.legend(['Max'])
plt.xlabel('$\Theta^2$ Cut')
plt.ylabel('$\chi$ Cut')
plt.xticks(range(th2_steps), [f'{value:.3f}' for value in th2_range], rotation=20)
plt.yticks(range(chi_steps), [f'{value:.2f}' for value in chi_range])
plt.title(
    f'Sensitivity: {sensitivity[chi_opt_idx, th2_opt_idx]:.2f} ± {error_sensitivity[chi_opt_idx, th2_opt_idx]:.2f} % of Crab')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{fig_folder}/sensitivity_grid_format_chi.png')
plt.savefig(f'{fig_folder}/sensitivity_grid_format_chi.pdf')
plt.close()

# %%

gammaness_steps = 10
th2_steps = 10

gammaness_range = np.linspace(0.94, 0.98, gammaness_steps)
th2_range = np.linspace(0.005, 0.01, th2_steps)
# %
significance_vect = np.zeros((gammaness_steps, th2_steps))

for i, gammaness_cut in enumerate(gammaness_range):
    for j, th2_cut in enumerate(th2_range):
        condition_cut_signal = np.logical_and(separation_gammaness.flatten() > gammaness_cut, theta_sq_limato < th2_cut)
        condition_cut_bg = np.logical_and(separation_gammaness.flatten() > gammaness_cut,
                                          theta_sq_off_1_limato < th2_cut)

        signal = np.sum(condition_cut_signal)
        background = np.sum(condition_cut_bg)

        significance = (signal - background) / np.sqrt(background)
        if signal - background > 10 & (signal - background > 0.05 * background):
            significance_vect[i, j] = significance
        if significance > 35:
            print(f'signal: {signal}, background: {background}')

# %
significance_vect_nonan = np.nan_to_num(significance_vect)
gammaness_opt_idx, th2_opt_idx = np.unravel_index(significance_vect_nonan.argmax(), significance_vect.shape)
gammaness_opt = gammaness_range[gammaness_opt_idx]
th2_opt = th2_range[th2_opt_idx]

print(f'Optimum cut in Gammaness: {gammaness_opt}')
print(f'Optimum cut in Th2: {th2_opt}')
print(f'Significance with these values: {significance_vect[chi_opt_idx, th2_opt_idx]}')

# %
fig, ax = plt.subplots()
plt.imshow(significance_vect)
plt.plot(th2_opt_idx, gammaness_opt_idx, 'xr', markersize=20)
plt.xlabel('$\Theta^2$ Cut')
plt.ylabel('Gammaness Cut')
plt.xticks(range(th2_steps), [f'{value:.3f}' for value in th2_range], rotation=20)
plt.yticks(range(gammaness_steps), [f'{value:.5f}' for value in gammaness_range])
plt.title('Significance on Crab')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{fig_folder}/significance_grid_linear.pdf')
plt.close()

# %%
chi_cut = 2.78
th2_cut = 0.01
condition_cut_signal = np.logical_and(chi_gammaness > chi_cut, theta_sq_limato < th2_cut)
condition_cut_bg_1 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_1_limato < th2_cut)
condition_cut_bg_2 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_2_limato < th2_cut)
condition_cut_bg_3 = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_3_limato < th2_cut)

# CUT IN ENERGY
# condition_cut_signal = np.logical_and(condition_cut_signal, energy_cut)
# condition_cut_bg_1 = np.logical_and(condition_cut_bg_1, energy_cut)
# condition_cut_bg_2 = np.logical_and(condition_cut_bg_2, energy_cut)
# condition_cut_bg_3 = np.logical_and(condition_cut_bg_3, energy_cut)

# Compute Signal
signal = np.sum(condition_cut_signal)
background_1 = np.sum(condition_cut_bg_1)
background_2 = np.sum(condition_cut_bg_2)
background_3 = np.sum(condition_cut_bg_3)

# Average background value
# background = background_3
background = (background_1 + background_2 + background_3) / 3

# %%
energies_crab = energy_hadrons[condition_cut_signal]
# %%
plt.figure()
plt.hist(energy_te, weights=1 / energy_te, bins=100)
plt.grid()
plt.ylabel('Log GeV')
plt.savefig('/home/emariott/software_magic/output_data/pictures/energy_point.png')
