import pickle

from keras.models import load_model

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator

# import matplotlib
#
# matplotlib.use('TkAgg')
# %%
# with open('/data/magic_data/clean_6_3punto5/crab/events_labels.pkl', 'rb') as f:
#     crabID, labels = pickle.load(f)

# crab_npy_path = glob('/home/emariott/magic_data/crab_clean10_5/npy_dump/*.npy')
# %%
with open('/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl',
          'rb') as f:
    big_df, evt_list = pickle.load(f)

# %%
labels = {ID: 0 for ID in evt_list}

# %%
# Load the data
BATCH_SIZE = 128
crab_generator = MAGIC_Generator(list_IDs=evt_list,
                                 labels=labels,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/home/emariott/magic_data/crab/crab_npy',
                                 include_time=True)

# %%
model = load_model(
    '/data4T/qualcosadiquellocheeranellahomeemariott/deepmagic/output_data/checkpoints/Tranfer_Ensemble_SE_InceptionV3_SingleDense_energy_from40_last6_nofreeze_dense64_adam4e-4.hdf5')
model.load_weights(
    '/data4T/qualcosadiquellocheeranellahomeemariott/deepmagic/output_data/swa_models/transfer-SE-inc-v3-snap_2019-03-19_10-57-34_SWA.h5')
net_name = 'transfer-SE-inc-v3-snap-lowLR_SWA'
# %
crab_energy = model.predict_generator(crab_generator, workers=8, verbose=1, use_multiprocessing=True)
# %%
# net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'/data4T/CNN4MAGIC/results/MC_classification/crab_reconstructions/crab_energy_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(crab_energy, f)
# %%
