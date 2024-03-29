import pickle

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2

def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
#%%
BATCH_SIZE = 128
big_df_crab, crab_evt_list = pickle_read(
    '/ssdraptor/magic_data/crab/crab_data/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl')
labels_crab = {ID: 0 for ID in crab_evt_list}  # Dummy
crab_generator = MAGIC_Generator(list_IDs=crab_evt_list,
                                 labels=labels_crab,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/ssdraptor/magic_data/crab/crab_data/crab_npy',
                                 include_time=True)

# %%
model = SEDenseNet121_position_l2()
weights_path = '/data4T/qualcosadiquellocheeranellahomeemariott/deepmagic/output_data/snapshots/SEDenseNet121_position_noclean_Gold_2019-02-25_01-37-25-10.h5'
model.load_weights(weights_path)
# #%%
net_name = 'SEDenseNet121_position_noclean_Gold_SNAP10_minimumValidation'
# model.save(f'/data/new_magic/output_data/checkpoints/{net_name}.hdf5')
# model = load_model(
#     '/home/emariott/software_magic/output_data/checkpoints/SE-121-Position-TransferEnsemble5-from59to63.hdf5')

model.compile('sgd', 'mse')

# %
y_pred_test = model.predict_generator(crab_generator, workers=8, verbose=1, use_multiprocessing=False)
# %%
dump_name = f'/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/position_prediction_crab_{net_name}.pkl'
pickle_dump(dump_name, y_pred_test)
# %%
import numpy as np
big_df = big_df_crab
true_coords_mm = np.transpose(np.array([big_df['xcoord_crab'].values, big_df['ycoord_crab'].values]))

# %%
import matplotlib.pyplot as plt

# def compute_theta(pos_true, pos_pred, en_bin, pos_in_mm=True, folder='', net_name='', plot=True):
#     if pos_in_mm:
#         pos_true = pos_true * 0.00337  # in deg
#         pos_pred = pos_pred * 0.00337  # in deg
#
#     num_events = pos_pred.shape[0]
#     theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)
#
#     hist_theta_sq, bins = np.histogram(theta_sq, bins=num_events)
#     hist_theta_sq_normed = hist_theta_sq / float(num_events)
#     cumsum_hist = np.cumsum(hist_theta_sq_normed)
#     angular_resolution = np.sqrt(bins[np.where(cumsum_hist > 0.68)[0][0]])
#     if not plot:
#         return angular_resolution
#
#     plt.figure()
#     plt.hist(theta_sq, bins=80, log=True)
#     plt.xlim([0, 0.4])
#     plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
#     plt.title(f'{net_name} Direction Reconstruction. Energy {en_bin}')
#     plt.xlabel(r'$\theta^2$')
#     plt.ylabel('Counts')
#     plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
#     plt.savefig(folder + '/' + net_name + '_angular.png')
#     # plt.savefig(folder + '/' + net_name + '_angular' + str(en_bin) + '.eps')
#
#     return angular_resolution
# %%
pos_pred = y_pred_test
pos_true = true_coords_mm[:pos_pred.shape[0], :]

# pos_true = pos_true[:y_pred.shape[0]]
pos_in_mm = True
if pos_in_mm:
    pos_true_deg = pos_true * 0.00337  # in deg
    pos_pred_deg = pos_pred * 0.00337  # in deg

num_events = pos_pred.shape[0]
theta_sq = np.sum((pos_true_deg - pos_pred_deg) ** 2, axis=1)
theta_sq_off_1 = np.sum((-pos_true_deg - pos_pred_deg) ** 2, axis=1)
# %%
theta_sq_off1 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)
theta_sq_off2 = ((pos_true_deg[:, 0] + pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] - pos_pred_deg[:, 1]) ** 2)
theta_sq_off3 = ((pos_true_deg[:, 0] - pos_pred_deg[:, 0]) ** 2) + ((pos_true_deg[:, 1] + pos_pred_deg[:, 1]) ** 2)

theta_sq_off_all = np.concatenate([theta_sq_off1, theta_sq_off2, theta_sq_off3])

pickle_dump('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_on.pkl', theta_sq)
pickle_dump('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_3_off.pkl', theta_sq_off_all)
# %%
test_fold = '/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/plots'
plt.figure()
plt.hist(theta_sq_off1, alpha=0.4, bins=500, label='off 1')
plt.hist(theta_sq_off2, alpha=0.4, bins=500, label='off 2')
plt.hist(theta_sq_off3, alpha=0.4, bins=500, label='off 3')
plt.xlim([0, 0.5])
plt.legend()
plt.savefig(f'{test_fold}/theta_off_histograms.png')

# %%
theta_sq_off_global = (theta_sq_off1 + theta_sq_off2 + theta_sq_off3) / 3
print(theta_sq_off_global.shape)
plt.figure()
plt.hist(theta_sq_off_global, alpha=0.4, bins=500)
plt.xlim([0, 0.5])
plt.savefig(f'{test_fold}/theta_off_global.png')
#%%
pickle_dump('/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/theta_sq_off_global.pkl', theta_sq_off_global)
# %%
# with open(
#         '/home/emariott/software_magic/output_data/reconstructions/crab_MobileNetV2_separation_10_5_notime_alpha1.pkl',
#         'rb') as f:
#     separation = pickle.load(f)

crab_gammaness = pickle_read(
    '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/computed_data/crab_separation_efficientNet_B3_last3_lin.pkl')
separation = crab_gammaness
# %%
print(len(separation), len(theta_sq))
theta_sq_limato = theta_sq[:len(separation)]
# %%
gammaness = 0.5
is_gamma = separation.flatten() > gammaness
print(np.sum(is_gamma))

theta_sq_gamma = theta_sq_limato[is_gamma]
# %%

plt.figure()
plt.hist(theta_sq_gamma, bins=800, log=True)
plt.xlim([0, 0.4])
# plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
plt.title(f'{net_name} Direction Reconstruction.')
plt.xlabel(r'$\theta^2$')
plt.ylabel('Counts')
# plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
plt.savefig(f'output_data/pictures/histogram_position_crab_{gammaness}_log.png')

plt.figure()
plt.hist(theta_sq_gamma, bins=800, log=False)
plt.xlim([0, 0.4])
# plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
plt.title(f'{net_name} Direction Reconstruction.')
plt.xlabel(r'$\theta^2$')
plt.ylabel('Counts')
# plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
plt.savefig(f'output_data/pictures/histogram_position_crab_no_{gammaness}_log.png')

# %%
from tqdm import tqdm

net_name = 'efficientNet_B3_last3_lin'
# dump_name = f'/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/crab_separation_{net_name}.pkl'
# with open(dump_name, 'rb') as f:
#     separation_gammaness = pickle.load(f)
separation_gammaness = crab_gammaness
num_bins = 50

theta_sq_limato = theta_sq[:len(separation_gammaness)]
theta_sq_off_1_limato = theta_sq_off_1[:len(separation_gammaness)]
gammaness_list = [1 - 10 ** (-2.1)]
for gammaness in tqdm(gammaness_list):
    is_gamma = separation_gammaness.flatten() > gammaness
    theta_sq_off_all = np.concatenate([theta_sq_off1[is_gamma], theta_sq_off2[is_gamma], theta_sq_off3[is_gamma]])

    print(np.sum(is_gamma))
    plt.figure()
    plt.hist(theta_sq_off_all, alpha=0.5, weights=1 / 3 * np.ones(len(theta_sq_off_all)),
             bins=np.linspace(0, 0.16, num_bins), log=False, label='$\Theta^2$ Off')
    plt.hist(theta_sq_limato[is_gamma], alpha=1, histtype='step', bins=np.linspace(0, 0.16, num_bins),
             log=False,
             color='C3', label='$\Theta^2$ Signal')
    plt.hist(theta_sq_limato[is_gamma], alpha=0.15, histtype='stepfilled', bins=np.linspace(0, 0.16, num_bins),
             log=False,
             color='C3')

    # plt.xlim([0, 0.01])
    plt.vlines(0.015, 0, 190, linestyles='-.', label='$\Theta^2$ Off', alpha=0.6)

    plt.ylim([0, 110])
    plt.xlabel('$\Theta^2$')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(f'$\Theta^2$ plot with gammaness: {gammaness:.3f}')
    plt.savefig(
        f'/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/plots/theta2_crab_{gammaness}_zoom.png')
    # plt.savefig(
    #     f'/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/plots/theta2_crab_{gammaness}_zoom.pdf')

    plt.close()

# %%

gammaness_list = [1 - 10 ** (-2.78)]
for gammaness in tqdm(gammaness_list):
    is_gamma = separation_gammaness.flatten() > gammaness
    theta_sq_off_all = np.concatenate([theta_sq_off1[is_gamma], theta_sq_off2[is_gamma], theta_sq_off3[is_gamma]])

    print(np.sum(is_gamma))
    plt.figure()
    plt.hist(theta_sq_off_all, alpha=0.5, weights=1 / 3 * np.ones(len(theta_sq_off_all)),
             bins=np.linspace(0, 0.5, num_bins), log=False, label='$\Theta^2$ Off')

    plt.hist(theta_sq_limato[is_gamma], alpha=1, histtype='step', bins=np.linspace(0, 0.5, num_bins),
             log=False,
             color='C3', label='$\Theta^2$ Signal')
    plt.hist(theta_sq_limato[is_gamma], alpha=0.15, histtype='stepfilled', bins=np.linspace(0, 0.5, num_bins),
             log=False,
             color='C3')

    # plt.xlim([0, 0.01])
    # plt.vlines(0.015, 0, 220, linestyles='-.', label='$\Theta^2$ Off', alpha=0.6)

    plt.ylim([0, 210])
    plt.xlabel('$\Theta^2$')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(f'$\Theta^2$ plot with gammaness: {gammaness:.3f}')
    plt.savefig(
        f'/home/emariott/software_magic/output_data/pictures/crab_significance_2/theta2_crab_{gammaness}.png')
    plt.savefig(
        f'/home/emariott/software_magic/output_data/pictures/crab_significance_2/theta2_crab_{gammaness}.pdf')

    plt.close()


# %%

plt.figure()
plt.hist2d(pos_pred[:, 0], pos_pred[:, 1], bins=550)
plt.plot(pos_true[1, 0], pos_true[1, 1], 'xr')
plt.plot(pos_true[-1, 0], pos_true[-1, 1], 'xk')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.legend(['True Crab Direction (Acquisition 1)', 'True Crab Direction (Acquisition 2)'])
# plt.colorbar()
plt.title('Heatmap of position predictions')
plt.savefig(f'/home/emariott/software_magic/output_data/pictures/hist2d_pos_pred_overlap.png')
plt.close()

# %%
gammaness = 0.8
gammaness_list = [0, 1 - 10 ** (-0.5), 1 - 10 ** (-1), 1 - 10 ** (-2), 1 - 10 ** (-2.78), 1 - 10 ** (-3)]
plt.figure(figsize=(10, 12))
for i, gammaness in enumerate(tqdm(gammaness_list)):
    is_gamma = separation_gammaness.flatten() > gammaness

    # plt.figure()
    plt.subplot(3, 2, i + 1)
    plt.hist2d(pos_pred_deg[is_gamma, 0], pos_pred_deg[is_gamma, 1], bins=np.linspace(-1.5, 1.5, 100))
    plt.plot(pos_true_deg[1, 0], pos_true_deg[1, 1], 'xr')
    if i < 4:
        plt.plot(pos_true_deg[-1, 0], pos_true_deg[-1, 1], 'xk')
    else:
        plt.plot(pos_true_deg[-1, 0], pos_true_deg[-1, 1], 'xm')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.legend(['True Crab Direction (Acquisition 1)', 'True Crab Direction (Acquisition 2)'])
    plt.xlabel('X camera coordinates (deg)')
    plt.ylabel('Y camera coordinates (deg)')


    plt.colorbar()
    plt.title(f'gammaness cut: {gammaness:.3f}')

plt.suptitle('Direction reconstruction of event triggered from the Crab Nebula', fontsize=20)
# plt.tight_layout()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(
    f'/home/emariott/software_magic/output_data/pictures/crab_significance_2/hist2d_pos_pred_gammaness_list_truepos.png')
plt.savefig(
    f'/home/emariott/software_magic/output_data/pictures/crab_significance_2/hist2d_pos_pred_gammaness_list_truepos.pdf')
plt.close()
# %%
plt.figure()
plt.hist2d(pos_true[:, 0], pos_true[:, 1], bins=150)
plt.savefig(f'/home/emariott/software_magic/output_data/pictures/hist2d_pos_true.png')
plt.close()

# %%
plt.figure()
plt.imshow(pos_pred)
plt.colorbar()
plt.savefig('/home/emariott/software_magic/output_data/pictures/position_pred_imshow.png')
