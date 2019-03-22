import pickle

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2

# %%
# from keras_generator import MAGIC_Generator

# from models import SEDenseNet121_position_l2

crabID_path = '/home/emariott/magic_data/crab/crab_npy'
# %%
from glob import glob

list = glob('/data/magic_data/clean_10_5/crab/npy_dump/*.npy')
print(len(list))
# %%
# crabID = [single_path[42:-4] for single_path in crabID_path]
# print(len(crabID))
with open('/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl',
          'rb') as f:
    big_df, evt_list = pickle.load(f)

# %%
labels = {ID: 1 for ID in evt_list}
# %%
# Load the data
BATCH_SIZE = 64
crab_generator = MAGIC_Generator(list_IDs=evt_list,
                                 labels=labels,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/home/emariott/magic_data/crab/crab_npy',
                                 include_time=True
                                 )
# %%
model = SEDenseNet121_position_l2()
weights_path = 'output_data/snapshots/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27-Best.h5'
model.load_weights(weights_path)
# #%%
net_name = 'SEDenseNet121_position_l2_fromEpoch41_best'
# model.save(f'/data/new_magic/output_data/checkpoints/{net_name}.hdf5')
# model = load_model(
#     '/home/emariott/software_magic/output_data/checkpoints/SE-121-Position-TransferEnsemble5-from59to63.hdf5')

model.compile('sgd', 'mse')

# %%
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

BATCH_SIZE = 64
machine = 'titanx'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    include_time=True,
    clean=False)

# %%
evalulation = model.evaluate_generator(val_gn, verbose=1, workers=8, use_multiprocessing=True)
print(evalulation)

# %%
y_pred_test = model.predict_generator(crab_generator, workers=8, verbose=1, use_multiprocessing=True)
# %%
net_name = 'SEDenseNet121_position_l2_fromEpoch41_best'

dump_name = f'/home/emariott/software_magic/output_data/reconstructions/position_prediction_crab_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)

# %%
with open(dump_name, 'rb') as f:
    y_pred = pickle.load(f)

# %%
import numpy as np

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
pos_true = true_coords_mm

pos_true = pos_true[:y_pred.shape[0]]
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
test_fold = '/home/emariott/software_magic/output_data/pictures/crab_theta2_hist/tests'
plt.figure()
plt.hist(theta_sq_off_1, alpha=0.4, bins=500)
plt.hist(theta_sq_off_2, alpha=0.4, bins=500)
plt.hist(theta_sq_off_3, alpha=0.4, bins=500)
plt.xlim([0, 0.5])
# plt.legend(['1','2','3'])
plt.savefig(f'{test_fold}/theta_off_2')

# %%
theta_sq_off_global = (theta_sq_off_1 + theta_sq_off_2 + theta_sq_off_3) / 3
print(theta_sq_off_global.shape)
plt.figure()
plt.hist(theta_sq_off_global, alpha=0.4, bins=500)
plt.xlim([0, 0.5])
# plt.legend(['1','2','3'])
plt.savefig(f'{test_fold}/theta_off_global')

# %%
with open(
        '/home/emariott/software_magic/output_data/reconstructions/crab_MobileNetV2_separation_10_5_notime_alpha1.pkl',
        'rb') as f:
    separation = pickle.load(f)

# separation = separation[:num_events]
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

net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/crab_separation_{net_name}.pkl'
with open(dump_name, 'rb') as f:
    separation_gammaness = pickle.load(f)

theta_sq_limato = theta_sq[:len(separation_gammaness)]
theta_sq_off_1_limato = theta_sq_off_1[:len(separation_gammaness)]
gammaness_list = [0.98878]
for gammaness in tqdm(gammaness_list):
    is_gamma = separation_gammaness.flatten() > gammaness
    print(np.sum(is_gamma))
    plt.figure()
    plt.hist(theta_sq_limato[is_gamma], alpha=0.5, bins=np.linspace(0, 0.1, 50), log=True)
    plt.hist(theta_sq_off_1_limato[is_gamma], alpha=0.5, bins=np.linspace(0, 0.1, 50), log=True)
    # plt.xlim([0, 0.01])
    plt.xlabel('$\Theta^2$')
    plt.ylabel('Counts')
    plt.legend(['Theta Signal', 'Theta Off'])
    plt.title(f'$\Theta^2$ plot with gammaness: {gammaness}')
    plt.savefig(
        f'/home/emariott/software_magic/output_data/pictures/crab_theta2_posok/linspace_log_ue_final_finegrained_lin_0.1_gammaness_{gammaness}_histogram_thetas_thetaoff.png')
    # plt.savefig(
    #     f'/home/emariott/software_magic/output_data/pictures/crab_theta2_posok/final_finegrained_lin_ye_0.1_gammaness_{gammaness}_histogram_thetas_thetaoff.pdf')

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
gammaness_list = [0, 0.5, 0.7, 0.9, 0.98878, 0.999]
plt.figure(figsize=(10, 12))
for i, gammaness in enumerate(tqdm(gammaness_list)):
    is_gamma = separation_gammaness.flatten() > gammaness

    # plt.figure()
    plt.subplot(3, 2, i + 1)
    plt.hist2d(pos_pred[is_gamma, 0], pos_pred[is_gamma, 1], bins=250)
    plt.plot(pos_true[1, 0], pos_true[1, 1], 'xr')
    if i < 4:
        plt.plot(pos_true[-1, 0], pos_true[-1, 1], 'xk')
    else:
        plt.plot(pos_true[-1, 0], pos_true[-1, 1], 'xm')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.legend(['True Crab Direction (Acquisition 1)', 'True Crab Direction (Acquisition 2)'])

    plt.colorbar()
    plt.title(f'gammaness cut: {gammaness}')

plt.suptitle('Direction reconstruction of event triggered from the Crab Nebula', fontsize=20)
# plt.tight_layout()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(
    f'/home/emariott/software_magic/output_data/pictures/Crab_position_hist_gammaness/hist2d_pos_pred_gammaness_list_truepos.png')
# plt.savefig(
#     f'/home/emariott/software_magic/output_data/pictures/Crab_position_hist_gammaness/hist2d_pos_pred_gammaness_list_truepos.pdf')
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
