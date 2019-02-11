import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

pic_folder = 'notebooks/pic_folders'

# %%
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


def read_pkl(filename):
    with open(filename, 'rb') as f:
        y_pred = pickle.load(f)
    print(f'loaded {filename}')
    return y_pred


sp_gm = '/home/emariott/deepmagic/output_data/reconstructions/gamma_separation_test_predict_MobileNetV2-separation-big.pkl'
separation_gamma = read_pkl(sp_gm)

energy_gamma_filename = '/home/emariott/deepmagic/output_data/reconstructions/MobileNetV2_2dense_energy_snap_whole_epoch5.pkl'
energy_gamma = read_pkl(energy_gamma_filename)

pos_gamma_filename = '/home/emariott/deepmagic/output_data/reconstructions/pos_predMobileNetV2_4dense_position-big-2.pkl'
position_gamm = read_pkl(pos_gamma_filename)
# %%

energy_hadrons = read_pkl(
    '/home/emariott/deepmagic/output_data/reconstructions/hadron_pred/y_pred_new_energy_hadrons.pkl')
gammaness_hadrons = read_pkl(
    '/home/emariott/deepmagic/output_data/reconstructions/hadron_pred/y_pred_new_gammaness.pkl')
position_hadrons = read_pkl(
    '/home/emariott/deepmagic/output_data/reconstructions/hadron_pred/y_pred_new_position_hadrons.pkl')
hadrons_complement = read_pkl(
    '/home/emariott/deepmagic/output_data/reconstructions/hadron_pred/new_root_complement.pkl')

# %%
# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_test = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                       want_golden=False,
                                                                       want_energy=True)

# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, position_test = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                         want_golden=False,
                                                                         want_position=True)

# %%
position_test_trimmed = position_test[:position_gamm.shape[0]]
energy_test_trimmed = energy_test[:energy_gamma.shape[0]]

# %%
###### Read files #######

e = energy_test_trimmed

# We take as triggered energies an array of 10% the size of the original
e_trig = e[separation_gamma.flatten() > 0.5]

# %%
# If want linar
e = np.power(10, e)
e_trig = np.power(10, e_trig)


Simulated_Events = np.size(e)
Triggered_Events = np.size(e_trig)
fig, ax = plt.subplots()
ax.hist(np.log10(e), bins=100)
ax.set_yscale("log")
plt.savefig(f'{pic_folder}/hist2.png')
plt.close()

# %%

##### Binnings and constants######
# Whenever implemented using simulated files, most of these values can be read from the simulations
emin = 50.  # GeV
emax = 30.e3  # GeV
eedges = 13
ebins = eedges - 1
E = np.logspace(math.log10(emin), math.log10(emax), eedges)
Emed = np.sqrt(E[:-1] * E[1:])

gammaness_bins = 10
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
            print(f'NSIM: {Nsim_w}')
            print(f'Ntrig_w: {Ntrig_w}')
            area.append(Ntrig_w / Nsim_w * Area_sim)  # cm^2

    return area


# %%


# Plot the collection area
area = collection_area(e, e_trig)
fig, ax = plt.subplots()
ax.set_ylim(1.e6, 4.e10)
ax.set_xlabel("Energy [GeV]")
ax.set_ylabel("Collection area [cm$^2$]")
ax.grid(ls='--', alpha=0.4)
ax.loglog(E[:-1], area)
plt.savefig(f'{pic_folder}/collection_area2.png')
plt.close()

# %%


# Fake energy, gammaness and theta2 arrays for simulated and triggered events
# (To be substituted by real np.array with all the simulated/triggered)

trigger = separation_gamma.flatten() > 0.5

gammaness = separation_gamma
gammaness_trig = separation_gamma[trigger]

pos_true = position_test_trimmed
pos_pred = position_gamm
pos_in_mm = True
if pos_in_mm:
    pos_true = pos_true * 0.00337  # in deg
    pos_pred = pos_pred * 0.00337  # in deg

theta2 = np.sum((pos_true - pos_pred) ** 2, axis=1)

theta2_trig = np.sum((pos_true[trigger] - pos_pred[trigger]) ** 2, axis=1)  # maximum 0.5 deg^2

trigger_hadron = gammaness_hadrons.flatten() > 0.5
# Same for hadrons
gammaness_h = gammaness_hadrons
gammaness_trig_h = gammaness_hadrons[trigger_hadron]

# %% ???? COME FACCIO QUA???????
# TODO: check theta2 for hadrons
# pos_pred_h = position_hadrons
# pos_in_mm=True
# if pos_in_mm:
#     pos_pred_h = pos_pred_h * 0.00337  # in deg
#
# theta2_h = np.sum((pos_true - pos_pred_h) ** 2, axis=1)
#
# theta2_trig_h = np.sum((pos_true[trigger] - pos_pred[trigger]) ** 2, axis=1)  # maximum 0.5 deg^2

# %%
##### ??????
# TODO: check time for hadrons
obs_time_h = np.sum(hadrons_complement[0]['timediff'])
####### Sensitivity calculation ##########
# We will first go for a implementation using Sig = Nex/sqrt(Nbg)
# obstime = 50 * 3600 # s (50 hours)


# %%
####### Weighting of the hadrons #####
# No simulation, just take the gamma energy distribution and convert it to hadrons

# Float_t ProtonTrueSpectralIndex = -2.70;
# Float_t ProtonTrueNorm = 9.6e-9;  // (cm2 sr s GeV)^-1 at ProtonEnorm
# Float_t ProtonEnorm    = 1000.; // GeV

K = Simulated_Events * (1 + Index_sim) / (emax ** (1 + Index_sim) - emin ** (1 + Index_sim))
Max_impact_protons = 500.e2  # cm
Area_sim = math.pi * math.pow(Max_impact_protons, 2)  # cm^2
cone = 5. * math.pi / 180  # 5 deg
if (cone == 0):
    Omega = 1
else:
    Omega = 2 * np.pi * (1 - np.cos(cone))

K_w = 9.6e-11  # GeV^-1 cm^-2 s^-1
index_w = -2.7
E0 = 1000.  # GeV

Int_e1_e2 = K * E0 ** Index_sim
Np_ = Int_e1_e2 * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (index_w + 1)
Rp = K_w * Area_sim * Omega * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (
        index_w + 1)  # Rate (in Hz)
print("The total rate of simulated proton events is %.1f Hz" % Rp)

# %%

###### Weighting of the gamma simulations #####

# HEGRA Crab
#  TF1* CrabFluxHEGRA = new TF1("CrabFluxHEGRA","[0]*pow(x/1000.,-[1])",50,80000);
#  CrabFluxHEGRA->SetParameter(0,2.83e-11);
#  CrabFluxHEGRA->SetParameter(1,2.62);

K = Simulated_Events * (1 + Index_sim) / (emax ** (1 + Index_sim) - emin ** (1 + Index_sim))
Area_sim = math.pi * math.pow(Max_impact, 2)  # cm^2
cone = 0
if (cone == 0):
    Omega = 1
else:
    Omega = 2 * np.pi * (1 - np.cos(cone))

K_w = 2.83e-11  # GeV^-1 cm^-2 s^-1
index_w = -2.62
E0 = 1000.  # GeV

Int_e1_e2 = K * E0 ** Index_sim
N_ = Int_e1_e2 * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (index_w + 1)
R = K_w * Area_sim * Omega * (emax ** (index_w + 1) - emin ** (index_w + 1)) / (E0 ** index_w) / (
        index_w + 1)  # Rate (in Hz)
print("The total rate of simulated gamma events is %.1f Hz" % R)

# %%

e_w = ((e / E0) ** (index_w - Index_sim)) * R / N_
e_trig_w = ((e_trig / E0) ** (index_w - Index_sim)) * R / N_
ep_w = ((e / E0) ** (index_w - Index_sim)) * Rp / Np_

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
ax2.hist(np.log10(e), histtype=u'step', bins=20, weights=ep_w, label="Protons")
ax2.legend()
ax2.set_yscale('log')
ax2.set_xlabel("$log_{10}E (GeV)$")
ax2.grid(ls='--', alpha=.5)
# plt.xscale('log')
plt.savefig(f'{pic_folder}/hist_non_so.png')
plt.close()

# %%

# Number of gammas per bin without any cuts in theta2 or gammaness
for i in range(0, ebins):  # binning in energy
    e_w_sum = np.sum(e_w[(e < E[i + 1]) & (e > E[i])])
    print("Rate of gammas between %.1f GeV and %.1f GeV: %.2f Hz" % (E[i], E[i + 1], e_w_sum))

# %%
# TODO: observation time of Gammas?
# TODO: Again: this fucking theth2_h...
# Cut optimization for gammas and hadrons

final_gamma = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))
final_hadrons = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))

for i in range(0, eedges - 1):  # binning in energy
    e_w_binE = np.sum(e_w[(e < E[i + 1]) & (e > E[i])])
    for g in range(0, gammaness_bins):  # cut in gammaness
        Ngammas = []
        Nhadrons = []
        for t in range(0, theta2_bins):  # cut in theta2
            e_trig_w_sum = np.sum(e_trig_w[(e_trig < E[i + 1]) & (e_trig > E[i]) \
                                           & (gammaness_trig > 0.1 * g) & (theta2_trig < 0.05 * (t + 1))])
            # Just considering all the hadrons give trigger...
            ep_w_sum = np.sum(ep_w[(e < E[i + 1]) & (e > E[i]) \
                                   & (gammaness_h > 0.1 * g) & (theta2_h < 0.05 * (t + 1))])

            final_gamma[i][g][t] = e_trig_w_sum * obstime
            final_hadrons[i][g][t] = ep_w_sum * obs_time_h


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
    significance = (Ng) / np.sqrt(Nh * alpha)
    # significance = significanceLiMa(Ng+Nh, Nh, alpha)
    sensitivity = 5 / significance * 100  # percentage of Crab

    return sensitivity


sens = Calculate_sensitivity(final_gamma, final_hadrons, 1)

# %%
Tolerance_abs = 0.1


def Calculate_sensitivity_LiMa(Ng, Nh, alpha):
    Non_loop = Ng + Nh
    Tolerance = 1
    # for i in range(0,100):
    while (Tolerance > Tolerance_abs):
        SignificanceLiMa_loop = significanceLiMa(Non_loop, Nh, alpha)
        Tolerance = abs(5 - SignificanceLiMa_loop)
        # print(Non_loop, Nh, SignificanceLiMa_loop, Tolerance)
        if (Tolerance < Tolerance_abs):
            break
        if ((SignificanceLiMa_loop > 5.) & (Non_loop - Nh > 0)):
            Non_loop = Non_loop - 0.005 * Non_loop
        else:
            Non_loop = Non_loop + 0.005 * Non_loop

    return abs(Non_loop - Nh) / Ng * 100


sens_LiMa = np.ndarray(shape=(ebins, gammaness_bins, theta2_bins))

for ix, iy, iz in np.ndindex(final_gamma.shape):
    # print(ix,iy,iz)
    sens_LiMa[ix, iy, iz] = Calculate_sensitivity_LiMa(final_gamma[ix, iy, iz], final_hadrons[ix, iy, iz], 1)
    # print(final_gamma[ix,iy,iz], final_hadrons[ix,iy,iz], final_gamma[ix,iy,iz] / final_hadrons[ix,iy,iz])
    # print(sens_LiMa[ix,iy,iz], sens[ix,iy,iz])
    # if (iy == 1):
    #    break


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
    pl = ax.imshow(sens_LiMa[ebin], cmap='viridis', extent=[0., 0.5, 1., 0.])
    fill_bin_content(ax, ebin)

    format_axes(ax, pl)
    plt.savefig(f'{pic_folder}/sensitiviti_bin_{ebin}.png')
    plt.close()


#########
#########
########
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
    cbar = fig.colorbar(pl, cax=cbaxes)
    cbar.set_label('Sensitivity (% Crab)', fontsize=15)


# fig, ax = plt.subplots(figsize=(8,8), )
fig, axarr = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(13.2, 18))
indices = []
sensitivity = np.ndarray(shape=ebins)

for ebin in range(0, ebins):
    arr_i = int(ebin / 3)
    arr_j = ebin - int(ebin / 3) * 3
    pl = axarr[arr_i, arr_j].imshow(sens_LiMa[ebin], cmap='viridis_r', extent=[0., 0.5, 1., 0.],
                                    # vmin=sens_LiMa.min(), vmax=sens_LiMa.max())
                                    norm=LogNorm(vmin=sens_LiMa.min(), vmax=sens_LiMa.max()))
    format_axes_array(axarr[arr_i, arr_j], arr_i, arr_j)

    # gammaness/theta2 indices where the minimum in sensitivity is reached
    ind = np.unravel_index(np.argmin(sens_LiMa[ebin], axis=None), sens_LiMa[ebin].shape)
    indices.append(ind)
    sensitivity[ebin] = sens_LiMa[ebin][ind]

fig.subplots_adjust(hspace=0, wspace=0)
# format_axes(ax)
plt.savefig(f'{pic_folder}/boh_non_ho_capito.png')
plt.close()


# %%
def Crab_spectrum(x):
    MAGIC_par = [3.23e-11, -2.47, -0.24]
    # dFdE = MAGIC_par[0]*pow(x/1.,MAGIC_par[1]+MAGIC_par[2]*np.log10(x/1.))
    dFdE = MAGIC_par[0] * pow(x / 1000., MAGIC_par[1] + MAGIC_par[2] * np.log10(x / 1000.))

    return dFdE


# %%
def plot_MAGIC_Sensitivity(ax):
    s = np.loadtxt('magic_sensitivity.txt')
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
    ax.set_xlim(5e1, 9.e4)
    ax.set_ylim(1.e-14, 5.e-10)
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
plt.savefig(f'{pic_folder}/crab_trallalla.png')
plt.close()
