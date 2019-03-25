import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# %%
pic_folder = '/home/emariott/software_magic/output_data/sensitivity'

# %%
# %
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


def read_pkl(filename):
    with open(filename, 'rb') as f:
        y_pred = pickle.load(f)
    print(f'loaded {filename}')
    return y_pred


pos_in_mm = True

sp_gm = '/home/emariott/software_magic/output_data/reconstructions/point_MobileNetV2_separation_10_5_notime_alpha1_DIOKKA_2.pkl'
separation_gamma = read_pkl(sp_gm)

energy_gamma_filename = '/home/emariott/software_magic/output_data/reconstructions/energy_titanx_transfer-SE-inc-v3-snap-lowLR_SWA.pkl'
energy_gamma = read_pkl(energy_gamma_filename)

pos_gamma_filename = '/home/emariott/software_magic/output_data/reconstructions/SE-121-Position-l2-fromepoch80_2019-03-17_23-13-18.pkl'
position_gamm = read_pkl(pos_gamma_filename)
# %

energy_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/crab_energy_transfer-SE-inc-v3-snap-lowLR_SWA.pkl')
gammaness_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/crab_separation_MobileNetV2_separation_10_5_notime_alpha1.pkl')
position_hadrons = read_pkl(
    '/home/emariott/software_magic/output_data/reconstructions/position_prediction_crab_SEDenseNet121_position_l2_fromEpoch41_best.pkl')
big_df_crab, evt_list_crab = read_pkl(
    '/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl')

energy_sim = read_pkl('/home/emariott/software_magic/output_data/for_sensitivity/energy_sim_test_point.pkl')

# %%
# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_test = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                       machine='titanx',
                                                                       want_golden=True,
                                                                       want_energy=True)

# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, position_test = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                         machine='titanx',
                                                                         want_golden=True,
                                                                         want_position=True)

# %%
# event_to_trim = np.min(position_gamm.shape[0], energy_gamma.shape[0])
position_test_trimmed = position_test[:position_gamm.shape[0]]
energy_test_trimmed = energy_test[:energy_gamma.shape[0]]
gammaness_trimmed = separation_gamma[:energy_gamma.shape[0]]
# %
print(len(energy_test_trimmed) - len(gammaness_trimmed))
# %% Select hadrons from crab
position_crab_mm = np.transpose(np.array([big_df_crab['xcoord_crab'].values, big_df_crab['ycoord_crab'].values]))
position_crab_mm_limato = position_crab_mm[:position_hadrons.shape[0], :]
if pos_in_mm:
    position_crab_deg = position_crab_mm_limato * 0.00337  # in deg
    position_hadrons_deg_tmp = position_hadrons * 0.00337  # in deg
    # position_test_trimmed = position_test_trimmed * 0.00337
    # position_gamm = position_gamm * 0.00337
    pos_in_mm = False
# %%
condition_hadron_distant_from_crab = np.sqrt(np.sum((position_hadrons_deg_tmp - position_crab_deg) ** 2, axis=1)) > 0.5

# %%
chi_gammas = -np.log10(1 - gammaness_trimmed)
chi_hadrons = -np.log10(1 - gammaness_hadrons)

# %%
print(np.max(chi_hadrons), np.min(chi_hadrons))
print(np.max(chi_gammas), np.min(chi_gammas))
print(np.logspace(-6, -1, 10))
# %%

e = energy_sim

# We take as triggered energies an array of 10% the size of the original
e_trig = np.power(10, energy_test_trimmed)

Simulated_Events = np.size(e)
Triggered_Events = np.size(e_trig)
fig, ax = plt.subplots()
ax.hist(np.log10(e))
ax.set_yscale("log")
plt.savefig(f'{pic_folder}/hist1.png')

# %%

#### Binnings and constants######
# Whenever implemented using simulated files, most of these values can be read from the simulations
emin = 10.  # GeV
emax = 30.e3  # GeV
eedges = 11
ebins = eedges - 1
# E = np.logspace(math.log10(emin), math.log10(emax), eedges)
emin_bin = 100.
emax_bin = 10000.
E = np.logspace(math.log10(emin_bin), math.log10(emax_bin), eedges)

Emed = np.sqrt(E[:-1] * E[1:])

gammaness_bins = 20
theta2_bins = 10

# Maximum impact parameter simulated for low Zd is 350 m, therefore Area_sim is pi*350^2
Max_impact = 350.e2  # in cm
Area_sim = math.pi * math.pow(Max_impact, 2)  # cm^2

# Weighting of the MC from -2.0 to -2.6
Index_sim = -1.6
Index_Crab = -2.62

# Only applicable for the simulation of diffuse gamma or protons
cone = 0.


# %%

##### Collection area calculation ######
def collection_area(Esim, Etrig):
    # Esim are all the simulated energies
    # Etrig are the energies after cuts
    area = []
    Nsim = np.power(Esim, Index_Crab - Index_sim)
    Ncuts = np.power(Etrig, Index_Crab - Index_sim)

    for i in range(0, ebins):
        Nsim_w = np.sum(Nsim[(Esim < E[i + 1]) & (Esim > E[i])])
        Ntrig_w = np.sum(Ncuts[(Etrig < E[i + 1]) & (Etrig > E[i])])
        if (Nsim_w == 0):
            print("You have not simulated any events in the energy range between %.3f GeV and %.3f GeV" % (
                E[i], E[i + 1]))
            area.append(0)
        else:
            area.append(Ntrig_w / Nsim_w * Area_sim)  # cm^2

    return area


# %%
# Plot the collection area
area = collection_area(e, e_trig)
fig, ax = plt.subplots()
# ax.set_ylim(1.e6,1.e9)
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Collection area [cm$^2$]")
ax.grid(ls='--', alpha=0.4)
ax.loglog(E[:-1], area)
plt.title('Collection Area')
plt.savefig(f'{pic_folder}/collection_area.png')

# %%

# (To be substituted by real np.array with all the simulated/triggered)

# gammaness = np.random.random_sample((int(Simulated_Events),))
gammaness_trig = gammaness_trimmed

pos_true = position_test_trimmed
pos_pred = position_gamm
pos_in_mm = True
if pos_in_mm:
    pos_true = pos_true * 0.00337  # in deg
    pos_pred = pos_pred * 0.00337  # in deg

theta2_trig = np.sum((pos_true - pos_pred) ** 2, axis=1)

# Same for hadrons
gammaness_trig_h = gammaness_hadrons

# %%

####### Sensitivity calculation ##########
# We will first go for a implementation using Sig = Nex/sqrt(Nbg)
obstime = 50 * 3600  # s (50 hours)

# %%
K = Simulated_Events * (1 + Index_sim) / (emax ** (1 + Index_sim) - emin ** (1 + Index_sim))
Area_sim = math.pi * math.pow(Max_impact, 2)  # cm^2
cone = 0
if (cone == 0):
    Omega = 1
else:
    Omega = 2 * np.pi * (1 - np.cos(cone))

K_w = 2.83e-14  # GeV^-1 cm^-2 s^-1
index_w = -2.62
E0 = 1000.  # GeV

Int_e1_e2 = K * E0 ** Index_sim
N_ = Int_e1_e2 * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (index_w + 1)
R = K_w * Area_sim * Omega * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (
        index_w + 1)  # Rate (in Hz)
print("The total rate of simulated gamma events is %.4f Hz" % (R))

# %%

e_w = ((e / E0) ** (index_w - Index_sim)) * R / N_
e_trig_w = ((e_trig / E0) ** (index_w - Index_sim)) * R / N_
# ep_w = ((e/E0)**(index_w-Index_sim))*Rp/Np_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.hist(np.log10(e), histtype=u'step', bins=20, density=1, label="Simulated")
ax1.hist(np.log10(e), histtype=u'step', bins=20, weights=e_w, density=1, label="Weighted to Crab")
ax1.set_yscale('log')
# plt.xscale('log')
ax1.set_xlabel("$log_{10}E (GeV)$")
ax1.grid(ls='--', alpha=.5)
ax1.legend()

# ax2.hist(np.log10(e),histtype=u'step',bins=20,label="Simulated rate")
ax2.hist(np.log10(e), histtype=u'step', bins=20, weights=e_w, label="Simulated rate weighted to Crab")
ax2.hist(np.log10(e_trig), histtype=u'step', bins=20, weights=e_trig_w, label="Triggered rate weighted to Crab")
# ax2.hist(np.log10(e),histtype=u'step',bins=20,weights = ep_w,label="Protons")
ax2.legend()
ax2.set_yscale('log')
ax2.set_xlabel("$log_{10}E (GeV)$")
ax2.grid(ls='--', alpha=.5)
# plt.xscale('log')

plt.savefig(f'{pic_folder}/simulated_non_so.png')

# %%

# Number of gammas per bin without any cuts in theta2 or gammaness
for i in range(0, ebins):  # binning in energy
    e_w_sum = np.sum(e_w[(e < E[i + 1]) & (e > E[i])])
    print("Rate of gammas between %.1f GeV and %.1f GeV: %.2f Hz" % (E[i], E[i + 1], e_w_sum))

# %%

# df_h = hadrons_complement[0]

time_obs_h = np.sum(big_df_crab['timediff'])  # TODO: check the if from the scripts

pos_in_mm = True
if pos_in_mm:
    pos_hadron_deg = position_hadrons * 0.00337  # in deg
theta_2_h = np.sum((pos_hadron_deg - np.zeros(pos_hadron_deg.shape)) ** 2, axis=1)

ring_condition_1 = np.logical_and(theta_2_h > (0.15) ** 2, theta_2_h < (0.545) ** 2)  # small ring: 0.3, 0.48
ring_condition = np.logical_and(ring_condition_1, condition_hadron_distant_from_crab)
theta_selected_hardon_in_sector = theta_2_h[ring_condition]
print(theta_selected_hardon_in_sector.shape)

def compute_N_tot_h(cut_g, cut_energy_low, cut_energy_high):
    gammaness_h_cut = gammaness_hadrons[ring_condition].flatten() >= cut_g

    # print(f'gammaness numeber: {np.sum(gammaness_h_cut)}')
    energy_h_cut_high = np.power(10, energy_hadrons[
        ring_condition].flatten()) < cut_energy_high  # Energy_hadrons is in Log10
    energy_h_cut_low = np.power(10, energy_hadrons[ring_condition].flatten()) > cut_energy_low

    # print(f'Num of ene_high: {np.sum(energy_h_cut_high)}')
    # print(f'Num of ene_low: {  np.sum(energy_h_cut_low)}')

    n_tot_h_condition = np.logical_and(gammaness_h_cut, energy_h_cut_high)
    # print(f'1st condition: {np.sum(n_tot_h_condition)}')
    n_tot_h_condition = np.logical_and(n_tot_h_condition, energy_h_cut_low)
    # print(f'2nd condition: {np.sum(n_tot_h_condition)}')

    N_tot_h = np.sum(n_tot_h_condition)
    return N_tot_h


# %%
for i in range(eedges - 1):
    print(compute_N_tot_h(0, E[i], E[i + 1]))

# %%
num_tot = np.array([compute_N_tot_h(0, E[i], E[i + 1]) for i in range(eedges - 1)])
print(np.sum(num_tot))

# %%
# Cut optimization for gammas and hadrons
e_estimated = np.power(10, energy_test_trimmed.flatten())  # TODO: ???? non ci andrebbe l'enrgia ricostruita?
# gammaness_trig_trimmed = gammaness_trig[]

# e_estimated = np.power(10, energy_gamma.flatten())
# e_estimated = e_estimated[:len(gammaness_trig)]
# %%
from tqdm import tqdm
final_gamma = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))
final_hadrons = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))

n_gamma_no_weight = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))
n_hadrons_no_weight = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))
theta2_trig_limato = theta2_trig[:gammaness_trig.shape[0]]
# E = np.logspace(math.log10(emin), math.log10(emax), eedges)
for i, energy_bin in enumerate(tqdm(np.linspace(emin, emax, ebins))):
    # for g_idx, gammaness_loop in enumerate(np.logspace(-7, -1, gammaness_bins)):
    for g_idx, gammaness_loop in enumerate(np.linspace(0.9, 0.99, gammaness_bins)):
        for t_idx, theta2_loop in enumerate(np.linspace(0.0005, 0.05, theta2_bins)):
            e_trig_w_sum = np.sum(e_trig_w[(e_estimated < E[i + 1]) & (e_estimated > E[i]) \
                                           & (gammaness_trig.flatten() > gammaness_loop) & (
                                                       theta2_trig_limato < theta2_loop)])

            e_trig_sum = np.size(e_trig[(e_trig < E[i + 1]) & (e_trig > E[i]) \
                                        & (gammaness_trig.flatten() > gammaness_loop) & (
                                                    theta2_trig_limato < theta2_loop)])

            n_gamma_no_weight[i][g_idx][t_idx] = e_trig_sum

            final_gamma[i][g_idx][t_idx] = e_trig_w_sum * obstime
            # TODO: Compute final_hadrons in my way
            N_tot_h = compute_N_tot_h(cut_g=gammaness_loop, cut_energy_low=E[i], cut_energy_high=E[i + 1])
            # print(
            #     f'N_tot_h: {N_tot_h}.\t cut gammaness > {gammaness_loop}.\t cut theta < {theta2_loop}\t energy bin: {np.sqrt(E[i]*E[i+1])}')
            theta2_tmp = theta2_loop
            N_h = theta2_tmp / ((0.545) ** 2 - (0.15) ** 2) * N_tot_h * (
                    50 * 3600) / time_obs_h  # TODO: check if thet2 = t
            final_hadrons[i][g_idx][t_idx] = N_h
            # final_hadrons[i][g][t] = ep_w_sum * obstime

            n_hadrons_no_weight[i][g_idx][t_idx] = N_tot_h


# for i in range(0, eedges - 1):  # binning in energy
#     # e_w_binE = np.sum(e_w[(e < E[i+1]) & (e > E[i])])
#     for g in range(0, gammaness_bins):  # cut in gammaness
#
#         for t in range(0, theta2_bins):  # cut in theta2
#             e_trig_w_sum = np.sum(e_trig_w[(e_estimated < E[i + 1]) & (e_estimated > E[i]) \
#                                            & (gammaness_trig.flatten() > 0.1 * g) & (theta2_trig < 0.005 * (t + 1))])
#
#             e_trig_sum = np.size(e_trig[(e_trig < E[i + 1]) & (e_trig > E[i]) \
#                                         & (gammaness_trig.flatten() > 0.1 * g) & (theta2_trig < 0.005 * (t + 1))])
#
#             n_gamma_no_weight[i][g][t] = e_trig_sum
#
#             final_gamma[i][g][t] = e_trig_w_sum * obstime
#             # TODO: Compute final_hadrons in my way
#             N_tot_h = compute_N_tot_h(cut_g=0.1 * g, cut_energy_low=E[i], cut_energy_high=E[i + 1])
#             print(
#                 f'N_tot_h: {N_tot_h}.\t cut gammaness > {0.1*g}.\t cut theta < {0.005 * (t+1)}\t energy bin: {np.sqrt(E[i]*E[i+1])}')
#             theta2_tmp = 0.005 * (t + 1)
#             N_h = theta2_tmp / ((0.53) ** 2 - (0.2) ** 2) * N_tot_h * (
#                     50 * 3600) / time_obs_h  # TODO: check if thet2 = t
#             final_hadrons[i][g][t] = N_h
#             # final_hadrons[i][g][t] = ep_w_sum * obstime
#
#             n_hadrons_no_weight[i][g][t] = N_tot_h


# %%
def significanceLiMa_array(g, b, alpha):
    s = g + b

    l = g * np.log(g / s * (alpha + 1) / alpha)
    m = b * np.log(b / s * (alpha + 1))

    return np.sqrt((l + m) * 2)


def significanceLiMa(g, b, alpha):
    if ((b == 0) & (g == 0)):
        return 0
    if (b == 0):
        b = np.nextafter(b, 1)  # Avoid passing an absolute 0
    if (g == 0):
        g = np.nextafter(g, 1)  # Avoid passing an absolute 0

    s = g + b

    if ((s < 0) or (alpha <= 0)):
        return -1

    l = g * np.log(g / s * (alpha + 1) / alpha)
    m = b * np.log(b / s * (alpha + 1))

    if (l + m < 0):
        return -1
    else:
        return np.sqrt((l + m) * 2)


# %%
def Calculate_sensitivity(Ng, Nh, alpha):
    significance = Ng / np.sqrt(Nh * alpha)
    # significance = significanceLiMa(Ng+Nh, Nh, alpha)
    sensitivity = 5 / significance * 100  # percentage of Crab

    return sensitivity


sens = Calculate_sensitivity(final_gamma, final_hadrons, 1)

print(sens)


# print(sens.shape)
# %% plotting fz

def fill_bin_content(ax, energy_bin):
    for i in range(0, gammaness_bins):
        for j in range(0, theta2_bins):
            text = ax.text((j + 0.5) * 0.05, (i + 0.5) * 0.1, "%.2f %%" % sens[energy_bin][i][j],
                           ha="center", va="center", color="w")
    return ax


def format_axes(ax, pl):
    ax.set_aspect(0.5)

    ax.set_ylabel(r'Gammaness', fontsize=15)
    ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    cbaxes = fig.add_axes([0.9, 0.125, 0.03, 0.755])
    cbar = fig.colorbar(pl, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize=15)


# Sensitivity plots for different Energy bins
for ebin in range(0, ebins):
    fig, ax = plt.subplots(figsize=(8, 8))
    # pl = ax.imshow(sens_LiMa[ebin], cmap='viridis', extent=[0., 0.5, 1., 0.])
    pl = ax.imshow(sens[ebin], cmap='viridis', extent=[0., 0.5, 1., 0.])
    fill_bin_content(ax, ebin)

    format_axes(ax, pl)
    plt.suptitle(f'Energy Bin: {np.sqrt(E[ebin]*E[ebin+1])}')
    plt.savefig(f'{pic_folder}/sensitivity_ebin_{ebin}.png')


# %%


def format_axes_array(ax, arr_i, arr_j):
    ax.set_aspect(0.5)
    if ((arr_i == 0) and (arr_j == 0)):
        ax.set_ylabel(r'Gammaness', fontsize=15)
    if ((arr_i == 3) and (arr_j == 2)):
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)', fontsize=15)

    starty, endy = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(endy, starty, 0.1)[::-1])
    startx, endx = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(startx, endx, 0.1))

    cbaxes = fig.add_axes([0.91, 0.125, 0.03, 0.755])
    # cbar = fig.colorbar(pl, cax=cbaxes)
    # cbar.set_label('Sensitivity (% Crab)', fontsize=15)


# fig, ax = plt.subplots(figsize=(8,8), )
fig, axarr = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(13.2, 18))
indices = []
sensitivity = np.ndarray(shape=ebins)

for ebin in range(0, ebins):
    arr_i = int(ebin / 3)
    arr_j = ebin - int(ebin / 3) * 3
    pl = axarr[arr_i, arr_j].imshow(sens[ebin], cmap='viridis_r', extent=[0., 0.5, 1., 0.],
                                    # vmin=sens_LiMa.min(), vmax=sens_LiMa.max())
                                    norm=LogNorm(vmin=sens.min(), vmax=sens.max()))
    format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j)

    # gammaness/theta2 indices where the minimum in sensitivity is reached
    ind = np.unravel_index(np.argmin(sens[ebin], axis=None), sens[ebin].shape)
    indices.append(ind)
    print(ind)
    sensitivity[ebin] = sens[ebin][ind]
    print(
        "Between %.1f GeV and %.1f GeV, gamma rate: %.3f/min (%i original gammas), hadron rate: %.3f/min (%i original hadrons)"
        % (E[ebin], E[ebin + 1], final_gamma[ebin][ind] / obstime * 60, n_gamma_no_weight[ebin][ind],
           final_hadrons[ebin][ind] / obstime * 60, n_hadrons_no_weight[ebin][ind]))

fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig(f'{pic_folder}/sensitivity_full_maybe.png')
# format_axes(ax)


# %%
# sensitivity = np.ones(shape=ebins) * 1e9
# g_sens_min = np.ones(shape=ebins) * 1e9
# t_sense_min = np.ones(shape=ebins) * 1e9
# from tqdm import tqdm
#
# bkg_rate_paper = [2.41, 0.54, 0.066, 0.027, 0.0133, 0.0059, 0.0027, 0.020, 0.0014, 0.0046]
# e_bins_min_paper = [100, 158, 251, 398, 631, 1000, 1585, 2512, 3981, 6310]
#
# t_sens = []
# all_rates = []
# for ebin in tqdm(range(ebins)):
#     if ebins != len(e_bins_min_paper):
#         print('Not the same number of energy bins')
#         raise ValueError
#     for g in range(gammaness_bins):
#
#         for t in range(theta2_bins):
#             if (sens[ebin][g][t] < sensitivity[ebin]):
#                 # ind = np.unravel_index(np.argmin(sens[ebin], axis=None), sens[ebin].shape)
#                 # print(n_hadrons_no_weight[ebin][g][t])
#                 # if (n_hadrons_no_weight[ebin][g][t] > 10):
#                 rate = final_hadrons[ebin][int(g)][int(t)] / obstime * 60
#                 all_rates.append(rate)
#
#                 # TODO: fai un if in cui entra se il rate è + o - il 20% del nominale dalla tabella colonna bkg-rate
#                 reasonable = (rate < (bkg_rate_paper[ebin] * 1.2)) & (rate > (bkg_rate_paper[ebin] * 0.8))
#                 if rate is reasonable:
#                     print('ok')
#                     sensitivity[ebin] = sens[ebin][g][t]
#                     g_sens_min[ebin] = g
#                     t_sense_min[ebin] = t
#
# print(sensitivity)
# print(g_sens_min)
# print(t_sense_min)

# %%
np.max(np.array(all_rates))

# %%
for ebin, g, t in zip(range(ebins), g_sens_min, t_sense_min):
    print(
        "Between %.1f GeV and %.1f GeV, gamma rate: %.3f/min (%i original gammas), hadron rate: %.3f/min (%i original hadrons)"
        % (
            E[ebin], E[ebin + 1], final_gamma[ebin][int(g)][int(t)] / obstime * 60,
            n_gamma_no_weight[ebin][int(g)][int(t)],
            final_hadrons[ebin][int(g)][int(t)] / obstime * 60, n_hadrons_no_weight[ebin][int(g)][int(t)]))


# %%

def Crab_spectrum(x):
    MAGIC_par = [3.23e-11, -2.47, -0.24]
    # dFdE = MAGIC_par[0]*pow(x/1.,MAGIC_par[1]+MAGIC_par[2]*np.log10(x/1.))
    dFdE = MAGIC_par[0] * pow(x / 1000., MAGIC_par[1] + MAGIC_par[2] * np.log10(x / 1000.))

    return dFdE


def plot_MAGIC_Sensitivity(ax):
    s = np.loadtxt('/home/emariott/software_magic/notebooks/magic_sensitivity.txt')
    ax.loglog(s[:, 0], s[:, 3] * np.power(s[:, 0] / 1.e3, 2), color='red', label='MAGIC (Aleksic et al. 2014)')

    return ax


def plot_Crab(ax, percentage=100, **kwargs):
    # factor is the percentage of Crab
    En = np.logspace(math.log10(100), math.log10(3.e4), 40)  # in TeV
    dFdE = percentage / 100. * Crab_spectrum(En)
    ax.loglog(En, dFdE * En / 1.e3 * En / 1.e3, color='gray', **kwargs)

    return ax


def format_axes(ax):
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    # ax.set_xlim(5e1,9.e4)
    # ax.set_ylim(1.e-14,5.e-10)
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(r'E$^2$ $\frac{\mathrm{dN}}{\mathrm{dE}}$ [TeV cm$^{-2}$ s$^{-1}$]')
    ax.grid(ls='--', alpha=.5)


def plot_sensitivity(ax):
    dFdE = Crab_spectrum(Emed)
    ax.loglog(Emed, sensitivity / 100 * dFdE * Emed / 1.e3 * Emed / 1.e3, label='Sensitivity CNN')


#### SENSITIVITY PLOT ######
fig, ax = plt.subplots()
plot_MAGIC_Sensitivity(ax)
plot_sensitivity(ax)

plot_Crab(ax, label=r'Crab')
# plot_Crab(ax,10,ls='dashed',label='10% Crab')
plot_Crab(ax, 1, ls='dotted', label='1% Crab')

format_axes(ax)
ax.legend(numpoints=1, prop={'size': 9}, ncol=2, loc='upper right')
plt.savefig(f'{pic_folder}/crab_comparison_THE_PLOT2.png')
