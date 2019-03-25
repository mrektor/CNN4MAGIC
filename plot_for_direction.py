import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

# %%

folder_fig = '/home/emariott/software_magic/output_data/pictures/for_direction'

df_0 = pd.read_csv(
    '/home/emariott/software_magic/output_data/csv_logs/SEDenseNet121_position_noclean_Gold_2019-02-25_01-37-25.csv')
# df_1 = pd.read_csv('/home/emariott/software_magic/output_data/csv_logs/SEDenseNet121_position_noclean_Gold_fromEpoch35_2019-03-04_17-03-30.csv')
df_2 = pd.read_csv(
    '/home/emariott/software_magic/output_data/csv_logs/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27.csv')
df_3 = pd.read_csv(
    '/home/emariott/software_magic/output_data/csv_logs/SEDenseNet121_position_l2_fromEpoch60_2019-03-20_19-32-10.csv')

# %%

df_list = [df_0, df_2, df_3]

loss_list = [df['loss'].values for df in df_list]
val_loss_list = [df['val_loss'].values for df in df_list]
lr_list = [df['lr'].values for df in df_list]
len_list = [len(df) for df in df_list]
# %%
loss_full = np.hstack(loss_list)
val_loss_full = np.hstack(val_loss_list)
lr_full = np.hstack(lr_list)
# %%
plt.figure()
plt.semilogy(loss_full)
plt.semilogy(val_loss_full)
plt.grid('both', linestyle='--')

plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.vlines(np.cumsum(len_list) - 1, 300, np.max(val_loss_full) + 1000, 'k', linestyles='-.', alpha=0.7)
plt.xticks(range(0, len(loss_full), 5))
plt.ylim([300, np.max(val_loss_full) + 1000])
plt.legend(['Train Loss', 'Validation Loss'])
plt.title('Optimization Result of SE DenseNet-121 for Direction')
plt.savefig(f'{folder_fig}/loss_SE-121.pdf')
plt.close()

# %%
plt.figure()
plt.plot(lr_full)
plt.savefig(f'{folder_fig}/lr_SE-121.png')
plt.close()

# %%

# %%
##########


BATCH_SIZE = 128
machine = 'titanx'

# Load the data
train_gn, val_gn, test_gn, position_te = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    include_time=True,
    clean=False)

train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    include_time=True,
    clean=False)


# %%

def bin_data_mask(data, num_bins, bins=None):
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), num_bins)
    binned_values = np.zeros(data.shape)
    bins_masks = []
    for i, bin in enumerate(bins):
        if i < bins.shape[0] - 1:
            mask = np.logical_and(data >= bins[i], data <= bins[i + 1])
            binned_values[mask] = bin
            bins_masks.append(mask)
    return binned_values, bins, bins_masks


def compute_theta(pos_true, pos_pred, en_bin, pos_in_mm=True, folder='', net_name='', plot=True):
    if pos_in_mm:
        pos_true = pos_true * 0.00337  # in deg
        pos_pred = pos_pred * 0.00337  # in deg

    num_events = pos_pred.shape[0]
    theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)

    hist_theta_sq, bins = np.histogram(theta_sq, bins=num_events)
    hist_theta_sq_normed = hist_theta_sq / float(num_events)
    cumsum_hist = np.cumsum(hist_theta_sq_normed)
    angular_resolution = np.sqrt(bins[np.where(cumsum_hist > 0.68)[0][0]])
    if not plot:
        return angular_resolution

    plt.figure()
    plt.hist(theta_sq, bins=80, log=True)
    plt.xlim([0, 0.4])
    plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
    plt.title(f'{net_name} Direction Reconstruction. Energy {en_bin}')
    plt.xlabel(r'$\theta^2$')
    plt.ylabel('Counts')
    plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
    plt.savefig(folder + '/' + net_name + '_angular_' + str(en_bin) + '.png')
    # plt.savefig(folder + '/' + net_name + '_angular' + str(en_bin) + '.eps')

    return angular_resolution


# %%
def plot_angular_resolution(position_true_list, position_prediction_list, energy_true,
                            fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction',
                            net_name='', makefigure=True):
    fig_width = 8
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    marker_set = ['<', '>', 'v', '^', 's']
    for j, position_prediction in enumerate(position_prediction_list):

        binned_values, bins, bins_masks = bin_data_mask(energy_true[j], 11)
        resolutions = []
        bin_medians = []
        position_true = position_true_list[j]
        for i, mask in enumerate(bins_masks):
            bin_pos = position_true[mask]
            bin_pred_pos = position_prediction[mask]
            bin_value = np.sqrt(bins[i] * bins[i + 1])
            res = compute_theta(bin_pos, bin_pred_pos, en_bin=bin_value, plot=False,
                                folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction/histograms')
            resolutions.append(res)
            bin_medians.append(bin_value)
        plt.semilogx(10 ** np.array(bin_medians[2:]), resolutions[2:], '-', marker=marker_set[j])

    state_of_the_art_theta = np.array([0.157, 0.135, 0.108, 0.095, 0.081, 0.073, 0.071, 0.067, 0.065, 0.062, 0.056])
    state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])

    plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')
    # plt.xlim([100, 10000])
    # plt.ylim([0, 0.175])

    plt.xlabel('Energy (GeV)')
    plt.ylabel('Angular Resolution')
    plt.title('68% Containment Angular Resolution of SE-DenseNet121')
    plt.legend(['SWA Taining II', 'SWA Training III', 'Minimum Validation', 'Minimum Validation No Time', 'SWA No Time',
                'MAGIC Standard Analysis'])
    plt.grid(which='both', linestyle='--')
    plt.savefig(fig_folder + '/angular_resolution_TOTALE_4.pdf')
    # plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    # plt.show()


# %%
# plt.figure()
appello_list.append(position_prediction)
appello_list.append(position_prediction_2)

# %%

position_te_limato_list = [position_te[:pred.shape[0], :] for pred in appello_list]
energy_te_limato_list = [energy_te[:pred.shape[0]] for pred in appello_list]
# net_name = 'SE DenseNet-121 III SWA'
plot_angular_resolution(position_te_limato_list, appello_list, energy_te_limato_list, net_name=net_name,
                        fig_folder=folder_fig, makefigure=False)

# %%

shape_list = [pred.shape for pred in appello_list]
shape_test = [test.shape for test in position_te_limato_list]
print(shape_list)
print(shape_test)


# %%
def plot_improvement(position_true_list, position_prediction_list, energy_true,
                     fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction',
                     net_name='', makefigure=True):
    fig_width = 9
    plt.figure(figsize=(fig_width, fig_width * 0.618))
    marker_set = ['<', '>', 'v', '^', 's']
    for j, position_prediction in enumerate(position_prediction_list):

        binned_values, bins, bins_masks = bin_data_mask(energy_true[j], 11)
        resolutions = []
        bin_medians = []
        position_true = position_true_list[j]
        for i, mask in enumerate(bins_masks):
            bin_pos = position_true[mask]
            bin_pred_pos = position_prediction[mask]
            bin_value = np.sqrt(bins[i] * bins[i + 1])
            res = compute_theta(bin_pos, bin_pred_pos, en_bin=bin_value, plot=False,
                                folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction/histograms')
            resolutions.append(res)
            bin_medians.append(bin_value)

        state_of_the_art_theta = np.array([0.157, 0.135, 0.108, 0.095, 0.081, 0.073, 0.071, 0.067, 0.065, 0.062, 0.056])
        state_of_the_art_energy = np.array([95, 150, 230, 378, 599, 949, 1504, 2383, 3777, 5986, 9487])
        # print(bin_medians)
        res_interp = np.interp(state_of_the_art_energy, 10 ** np.array(bin_medians), resolutions)
        enhancement = 100 * (state_of_the_art_theta - res_interp) / state_of_the_art_theta
        # print(len(bin_medians))
        # print(len(resolutions))
        # print(len(res_interp))
        # print(len(enhancement))
        print(enhancement)
        plt.semilogx(np.array(state_of_the_art_energy), enhancement, '-', marker=marker_set[j])

    # plt.semilogx(state_of_the_art_energy, state_of_the_art_theta, '--*')
    # plt.xlim([100, 10000])
    # plt.ylim([0, 0.175])
    plt.hlines(0, 60, 10000, linestyles='-.')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Enhancement (%)')
    plt.title('Enhancement of Angular Resolution SE-DenseNet121 w.r.t Aleksic et al. (2016)')
    plt.legend(['No-Time Best', 'SWA Training III', 'Minimum Validation'])
    plt.grid(which='both', linestyle='--')
    plt.savefig(fig_folder + '/enhancement_TOTALE_2.png')
    # plt.savefig(fig_folder + '/angular_resolution' + net_name + '.eps')
    # plt.show()


# %%
appello_list = [position_prediction]
position_te_limato_list = [position_te[:pred.shape[0], :] for pred in appello_list]
energy_te_limato_list = [energy_te[:pred.shape[0]] for pred in appello_list]
plot_improvement(position_te_limato_list, appello_list, energy_te_limato_list, net_name=net_name,
                 fig_folder=folder_fig, makefigure=False)
