import glob
import pickle

import numpy as np

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


def pkl_load(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


# %%

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

# %%
direction_reco_filepath = glob.glob('output_data/reconstructions/*position*.pkl')
print(direction_reco_filepath)
# %
networks = [path[28:-4] for path in direction_reco_filepath]
# %
print(networks)

# %%
direction_reco_filepath = ['output_data/reconstructions/position_SEDenseNet121_position_l2_fromEpoch41_SWA_15last.pkl',
                           # '/home/emariott/deepmagic/output_data/reconstructions/SE-121-Position-l2-fromepoch80_2019-03-17_23-13-18.pkl',
                           # 'output_data/reconstructions/pos_predMobileNetV2_4dense_position-big-2.pkl',
                           'output_data/reconstructions/SEDenseNet121_position_l2_fromEpoch60_2019-03-20_19-32-10.pkl',
                           # 'output_data/reconstructions/pred_MV2-4D-30E-l2-EnsLast9_position.pkl',
                           'output_data/reconstructions/position_SE-DenseNet121_pos_gold_noclean_best.pkl',
                           'output_data/reconstructions/position_SEDenseNet121_position_l2_NOTIME-Best.pkl',
                           'output_data/reconstructions/position_SEDenseNet121_position_l2_NOTIME-SWA.pkl']
networks = ['position_SEDenseNet121_position_l2_fromEpoch41_SWA_15last',
            # 'SE-121-Position-l2-fromepoch80_2019-03-17_23-13-18',
            # 'pos_predMobileNetV2_4dense_position-big-2',
            'SEDenseNet121_position_l2_fromEpoch60_2019-03-20_19-32-10',
            # 'pred_MV2-4D-30E-l2-EnsLast9_position',
            'position_SE-DenseNet121_pos_gold_noclean_best',
            'no time best',
            'no time swa'
            ]

# direction_reco_filepath = ['/home/emariott/software_magic/output_data/reconstructions/SE-121-Position-l2-fromepoch80_2019-03-17_23-13-18.pkl']
# networks = [' fromepoch80, forse SWA']
# %%
# direction_reco_filepath.append(
#     '/home/emariott/deepmagic/output_data/reconstructions/transfer-SE-inc-v3-snap_2019-03-19_10-57-34.pkl')
# networks.append('transfer ens snap')
#
# direction_reco_filepath.append(
#     '/home/emariott/deepmagic/output_data/reconstructions/transfer-SE-inc-v3-snap-LR_0_05HIGH_2019-03-20_01-50-12.pkl')
# networks.append('transfer ens snap HIGHLR SWA')
#
# direction_reco_filepath.append(
#     '/home/emariott/deepmagic/output_data/reconstructions/energy_transfer-SE-inc-v3-snap-LR_0_05HIGH_Best.pkl')
# networks.append('transfer ens snap HIGHLR BEST')
# %%
appello = {net: pkl_load(net_path) for net_path, net in zip(direction_reco_filepath, networks)}
appello_list = [pkl_load(net_path) for net_path in direction_reco_filepath]
# %%
print(appello)

# %%
lengths = [pred.shape for pred in appello.values()]
print(lengths)
print(position_te.shape)


# %%
def compute_loss(y_pred):
    # print(len(y_pred), len(energy))
    direction_limato = position_te[:y_pred.shape[0], :]
    mse = np.mean((direction_limato - y_pred) ** 2)
    return mse


# %%
print(compute_loss(position_prediction_3))
# %%
losses_dict = {net: compute_loss(appello[net]) for net in networks}

print(losses_dict)
# %%
losses_list = [compute_loss(appello[net]) for net in networks]
# %%
import pandas as pd

# %%
# df = pd.DataFrame(losses, index=losses.keys())
# print(df)
# %%
fold_fig = '/home/emariott/deepmagic/output_data/pictures/pagelle'
import matplotlib.pyplot as plt

# %%
plt.figure(figsize=(20, 6))
plt.barh(networks, losses_list, log=False)
plt.tight_layout(h_pad=10)
plt.savefig(f'{fold_fig}/Inceptions.png')

# %%
df_swa_from60 = pd.read_csv(
    '/home/emariott/deepmagic/output_data/csv_logs/SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09.csv')

# %%
# import seaborn as sns

# sns.set()
plt.figure()
plt.plot(df_swa_from60['epoch'] + 20, df_swa_from60['loss'])
plt.plot(df_swa_from60['epoch'] + 20, df_swa_from60['val_loss'])
plt.hlines(losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'], 0 + 20, 9 + 20, colors='g',
           linestyles='dashdot')
plt.hlines(losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'], 0 + 20, 9 + 20,
           colors='r', linestyles='dashed')
plt.legend(['Loss on Train', 'Loss on Validation', 'Test Loss of Best Model', 'Test Loss of SWA'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.xticks(range(20, 30, 1))
# plt.grid(which='both')
plt.title('Optimization Results of SE-Inception-v3 Single Dense')

plt.savefig(f'{fold_fig}/training_ensembles_no_searborn_from20.png')

# %%

from_epoch = 0
df_swa_from40 = pd.read_csv(
    '/home/emariott/deepmagic/output_data/csv_logs/SE_InceptionV3_SingleDense_energy_yesTime_from40_2019-03-17_15-35-17.csv')

plt.figure()
plt.plot(df_swa_from40['epoch'] + from_epoch, df_swa_from40['loss'])
plt.plot(df_swa_from40['epoch'] + from_epoch, df_swa_from40['val_loss'])
plt.hlines(losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'], 10 + from_epoch, 20 + from_epoch,
           colors='g',
           linestyles='dashdot')
plt.hlines(losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'], 10 + from_epoch,
           20 + from_epoch,
           colors='r', linestyles='dashed')
plt.legend(['Loss on Train', 'Loss on Validation', 'Test Loss of Best Model', 'Test Loss of SWA'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.xticks(range(from_epoch, from_epoch + 20, 1))
# plt.grid(which='both')
plt.title('Optimization Results of SE-Inception-v3 Single Dense')

plt.savefig(f'{fold_fig}/training_ensembles_no_searborn_from_{from_epoch}.png')

# %%

from_epoch = 0
df_swa_from0 = pd.read_csv(
    '/home/emariott/deepmagic/output_data/csv_logs/SE_InceptionV3_SingleDense_energy_yesTime_2019-03-16_19-49-37.csv')

plt.figure()
plt.plot(df_swa_from0['epoch'] + from_epoch, df_swa_from0['loss'])
plt.plot(df_swa_from0['epoch'] + from_epoch, df_swa_from0['val_loss'])
plt.hlines(losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'], 10 + from_epoch, 20 + from_epoch,
           colors='g',
           linestyles='dashdot')
plt.hlines(losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'], 10 + from_epoch,
           20 + from_epoch,
           colors='r', linestyles='dashed')
plt.legend(['Loss on Train', 'Loss on Validation', 'Test Loss of Best Model', 'Test Loss of SWA'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
# plt.xticks(range(from_epoch,from_epoch+20,1))
# plt.grid(which='both')
plt.title('Optimization Results of SE-Inception-v3 Single Dense')

plt.savefig(f'{fold_fig}/training_ensembles_no_searborn_from_{from_epoch}.png')

# %%

full_loss = np.concatenate((df_swa_from40['loss'].values, df_swa_from60['loss'].values))
full_val_loss = np.concatenate((df_swa_from40['val_loss'].values, df_swa_from60['val_loss'].values))
num_epochs = full_loss.shape[0]

plt.figure()
plt.plot(range(0, num_epochs, 1), full_loss)
plt.plot(range(0, num_epochs, 1), full_val_loss)
plt.hlines(losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'], num_epochs - 10, num_epochs,
           colors='g',
           linestyles='dashdot')
plt.hlines(losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'], num_epochs - 10,
           num_epochs,
           colors='r', linestyles='dashed')
plt.legend(['Loss on Train', 'Loss on Validation', 'Test Loss of Best Model', 'Test Loss of SWA'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
# plt.xticks(range(from_epoch,from_epoch+20,1))
# plt.grid(which='both')
plt.title('Optimization Results of SE-Inception-v3 Single Dense')

plt.savefig(f'{fold_fig}/training_ensembles_no_searborn_global.png')

# %%
fold_fig = '/home/emariott/deepmagic/output_data/pictures'
full_loss = np.concatenate((df_swa_from0['loss'], df_swa_from40['loss'].values, df_swa_from60['loss'].values))
full_val_loss = np.concatenate(
    (df_swa_from0['val_loss'], df_swa_from40['val_loss'].values, df_swa_from60['val_loss'].values))
num_epochs = full_loss.shape[0]

plt.figure()
plt.plot(range(1, num_epochs + 1, 1), full_loss)
plt.plot(range(1, num_epochs + 1, 1), full_val_loss)
# plt.hlines(losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'], num_epochs - 10, num_epochs,
#            colors='g',
#            linestyles='dashdot')
# plt.hlines(losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'], num_epochs - 10, num_epochs,
#            colors='r', linestyles='dashed')
plt.legend(['Loss on Train', 'Loss on Validation'])  # , 'Test Loss of Best Model', 'Test Loss of SWA'])

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.xticks([1, 40, 60, 70])
plt.grid(which='both', linestyle='--')
plt.title('Optimization Results of SE-Inception-v3 Single Dense')

plt.savefig(f'{fold_fig}/training_ensembles_no_searborn_global_onlytrain.pdf')

# %%
fold_fig = '/home/emariott/deepmagic/output_data/pictures/for_paper'

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure()
plt.barh(['Minimum Validation Snapshot', 'SWA of last 10 Snapshots', 'TSE-SWA (low LR)', 'TSE-SWA (high LR)',
          'TSE Best (high LR)'],
         [losses_dict_male['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'],
          losses_dict_male['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09'],
          losses_dict_male['transfer ens snap'],
          losses_dict_male['transfer ens snap HIGHLR SWA'],
          losses_dict_male['transfer ens snap HIGHLR BEST']
          ], color=colors[:5])
# plt.ylim((0.150,0.20))
plt.tight_layout(rect=[0, 0.03, 1, 0.92])
# plt.grid()
plt.xlim([0.185, 0.25])
plt.xlabel('Mean Absolute Linear Error')
plt.title('Test Losses of SE-Inception-v3 Single Dense models')
plt.savefig(f'{fold_fig}/ensemble_results.png')
plt.savefig(f'{fold_fig}/ensemble_results.pdf')

# %%
ens_loss = [losses_dict['energy_SE_InceptionV3_SingleDense_energy_yestime_Best'],
            losses_dict['SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09']]
plt.figure()
plt.plot([0, 1], ens_loss, '.-')
plt.ylim([0, 0.020])
plt.xticks([0, 1])
plt.savefig(f'{fold_fig}/ensemble_results_point.png')
