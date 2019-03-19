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
dump_name = f'output_data/reconstructions/position_prediction_crab_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)
