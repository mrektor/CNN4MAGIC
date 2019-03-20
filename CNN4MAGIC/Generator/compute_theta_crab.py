import pickle

# from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2
from keras.models import load_model
# %%
from keras_generator import MAGIC_Generator

# from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
# from models import SEDenseNet121_position_l2

crabID_path = '/data/magic_data/crab_npy'  # glob('/data/magic_data/crab_npy/*.npy')
# %%
# crabID = [single_path[42:-4] for single_path in crabID_path]
# print(len(crabID))
with open('/data/magic_data/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl', 'rb') as f:
    big_df, evt_list = pickle.load(f)

# %%
labels = {ID: 1 for ID in evt_list}
# %%
# Load the data
BATCH_SIZE = 256
crab_generator = MAGIC_Generator(list_IDs=evt_list,
                                 labels=labels,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/data/magic_data/crab_npy',
                                 include_time=True
                                 )
# %%
# model = SEDenseNet121_position_l2()
# weights_path = 'output_data/snapshots/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27-Best.h5'
# model.load_weights(weights_path)
# #%%
# net_name = 'SEDenseNet121_position_l2_fromEpoch41_best'
# model.save(f'/data/new_magic/output_data/checkpoints/{net_name}.hdf5')
model = load_model('/data/new_magic/output_data/checkpoints/SEDenseNet121_position_l2_fromEpoch41_best.hdf5')

model.compile('sgd', 'mse')
# %%
y_pred_test = model.predict_generator(crab_generator, workers=24, verbose=1, use_multiprocessing=True)
# %
net_name = 'SEDenseNet121_position_l2_fromEpoch41_best'
dump_name = f'output_data/reconstructions/position_prediction_crab_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)

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

pos_in_mm = True
if pos_in_mm:
    pos_true = pos_true * 0.00337  # in deg
    pos_pred = pos_pred * 0.00337  # in deg

num_events = pos_pred.shape[0]
theta_sq = np.sum((pos_true - pos_pred) ** 2, axis=1)

plt.figure()
plt.hist(theta_sq, bins=80, log=True)
plt.xlim([0, 0.4])
# plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
plt.title(f'{net_name} Direction Reconstruction.')
plt.xlabel(r'$\theta^2$')
plt.ylabel('Counts')
# plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
plt.savefig(f'/data/new_magic/output_data/pictures/histogram_position_crab_log.png')

plt.figure()
plt.hist(theta_sq, bins=80, log=False)
plt.xlim([0, 0.4])
# plt.axvline(x=angular_resolution, color='darkorange', linestyle='--')
plt.title(f'{net_name} Direction Reconstruction.')
plt.xlabel(r'$\theta^2$')
plt.ylabel('Counts')
# plt.legend(['Angular Resolution: {:02e}'.format(angular_resolution)])
plt.savefig(f'/data/new_magic/output_data/pictures/histogram_position_crab_no_log.png')
