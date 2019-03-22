import pickle

import numpy as np

net_name_pos = 'SEDenseNet121_position_l2_fromEpoch41_best'
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

# %%

pos_pred = pos_crab
true_coords_mm = np.transpose(np.array([big_df['xcoord_crab'].values, big_df['ycoord_crab'].values]))
pos_true = true_coords_mm

pos_true = pos_true[:pos_pred.shape[0]]
pos_in_mm = True
if pos_in_mm:
    pos_true = pos_true * 0.00337  # in deg
    pos_pred = pos_pred * 0.00337  # in deg

num_events = pos_pred.shape[0]
theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)
theta_sq_off_1 = np.sum((-pos_true - pos_pred) ** 2, axis=1)
# %%
theta_sq_off_1 = ((pos_true[:, 0] + pos_pred[:, 0]) ** 2) + ((pos_true[:, 1] + pos_pred[:, 1]) ** 2)
theta_sq_off_2 = ((pos_true[:, 0] + pos_pred[:, 0]) ** 2) + ((pos_true[:, 1] - pos_pred[:, 1]) ** 2)
theta_sq_off_3 = ((pos_true[:, 0] - pos_pred[:, 0]) ** 2) + ((pos_true[:, 1] + pos_pred[:, 1]) ** 2)

# %%

theta_sq_limato = theta_sq[:len(separation_gammaness)]
theta_sq_off_1_limato = theta_sq_off_1[:len(separation_gammaness)]

# %%
chi_gammaness = -np.log10(1 - separation_gammaness + 1e-20).flatten()
# %%
import matplotlib.pyplot as plt

fig_folder = '/home/emariott/software_magic/output_data/pictures/crab_optim'
plt.figure()
plt.hist(chi_gammaness, log=True, bins=200)
plt.savefig(f'{fig_folder}/chi_gammaness.png')
plt.close()

# %%
chi_steps = 5
th2_steps = 10

chi_range = np.linspace(1, 4.8, chi_steps)
th2_range = np.linspace(0.001, 0.1, th2_steps)

significance_vect = np.zeros((chi_steps, th2_steps))

for i, chi_cut in enumerate(chi_range):
    for j, th2_cut in enumerate(th2_range):
        # condition_cut = np.logical_and(chi_gammaness < chi_cut, theta_sq_limato < th2_cut)

        # signal = np.sum(theta_sq_limato[condition_cut])
        # background = np.sum(theta_sq_off_1[condition_cut])
        # signal = len(theta_sq_limato[condition_cut])
        # background = len()

        condition_cut_signal = np.logical_and(chi_gammaness > chi_cut, theta_sq_limato < th2_cut)
        condition_cut_bg = np.logical_and(chi_gammaness > chi_cut, theta_sq_off_1_limato < th2_cut)

        signal = np.sum(condition_cut_signal)
        background = np.sum(condition_cut_bg)

        significance = (signal - background) / np.sqrt(background)
        if background > 3:
            significance_vect[i, j] = significance
        if significance > 35:
            print(f'signal: {signal}, background: {background}')

significance_vect_nonan = np.nan_to_num(significance_vect)
chi_opt_idx, th2_opt_idx = np.unravel_index(significance_vect_nonan.argmax(), significance_vect.shape)
chi_opt = chi_range[chi_opt_idx]
th2_opt = th2_range[th2_opt_idx]

print(f'Optimum cut in Chi: {chi_opt}')
print(f'Optimum cut in Th2: {th2_opt}')
print(f'Significance with these values: {significance_vect[chi_opt_idx, th2_opt_idx]}')
# %
fig, ax = plt.subplots()
plt.imshow(significance_vect)
plt.xlabel('$\Theta^2$ Cut')
plt.ylabel('$\chi$ Cut')
plt.xticks(range(th2_steps), [f'{value:.3f}' for value in th2_range], rotation=20)
plt.yticks(range(chi_steps), [f'{value:.2f}' for value in chi_range])
plt.title('Significance on Crab')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{fig_folder}/significance_grid_format_chi.png')
plt.savefig(f'{fig_folder}/significance_grid_format_chi.pdf')
plt.close()

# %%

gammaness_steps = 5
th2_steps = 10

gammaness_range = np.linspace(0.98878, 1, gammaness_steps)
th2_range = np.linspace(0.001, 0.1, th2_steps)
# %
significance_vect = np.zeros((gammaness_steps, th2_steps))

for i, gammaness_cut in enumerate(gammaness_range):
    for j, th2_cut in enumerate(th2_range):
        condition_cut_signal = np.logical_and(separation_gammaness.flatten() < gammaness_cut, theta_sq_limato < th2_cut)
        condition_cut_bg = np.logical_and(separation_gammaness.flatten() < gammaness_cut,
                                          theta_sq_off_1_limato < th2_cut)

        signal = np.sum(condition_cut_signal)
        background = np.sum(condition_cut_bg)

        significance = (signal - background) / np.sqrt(background)

        significance_vect[i, j] = significance

# %
significance_vect_nonan = np.nan_to_num(significance_vect)
gammaness_opt_idx, th2_opt_idx = np.unravel_index(significance_vect_nonan.argmax(), significance_vect.shape)
gammaness_opt = gammaness_range[gammaness_opt_idx]
th2_opt = th2_range[th2_opt_idx]

print(f'Optimum cut in Gammaness: {gammaness_opt}')
print(f'Optimum cut in Th2: {th2_opt}')

# %
fig, ax = plt.subplots()
plt.imshow(significance_vect)
plt.xlabel('$\Theta^2$ Cut')
plt.ylabel('Gammaness Cut')
plt.xticks(range(th2_steps), [f'{value:.3f}' for value in th2_range], rotation=20)
plt.yticks(range(gammaness_steps), [f'{value:.5f}' for value in gammaness_range])
plt.title('Significance on Crab')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{fig_folder}/significance_grid_linear.png')
plt.close()
