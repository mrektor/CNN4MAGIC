import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SEDenseNet121_position
from keras.layers import Input, Dense, concatenate, BatchNormalization, LeakyReLU
from keras.models import Model
from glob import glob
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

# %%
BATCH_SIZE = 16
machine = 'towerino'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    clean=False)
# %

snap_folder = '/home/emariott/deepmagic/output_data/snapshots'
# %%
snapshots = glob(f'{snap_folder}/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27-*.h5')
snapshots_sort = sorted(snapshots)

# %
print(snapshots_sort)
print(len(snapshots_sort))

# %%
freeze = True

input_img = Input(shape=(67, 68, 4), name='ensemble_input')
feature_layers = []
for idx, snap in enumerate(tqdm(snapshots_sort)):
    model = SEDenseNet121_position(input_img)
    for single_layer in model.layers:
        tmp_name = single_layer.name
        single_layer.name = single_layer.name + f'_snap_{idx}'
    if freeze:
        for single_layer in model.layers:
            single_layer.trainable = False
    model.load_weights(snap)
    feature_layers.append(model.layers[-2].output)

one_layer = concatenate(feature_layers)

out_ens = Dense(80, kernel_regularizer='l2')(one_layer)
out_ens = BatchNormalization()(out_ens)
out_ens = LeakyReLU()(out_ens)
out_ens = Dense(2, kernel_regularizer='l2')(out_ens)

ensemble_model = Model(input_img, out_ens)

# %%
ensemble_model.summary()

# %%
net_name = 'SE-121-Position-TransferEnsemble5-from59to63'
check = ModelCheckpoint(f'/home/emariott/deepmagic/output_data/checkpoints/{net_name}.hdf5', save_best_only=True)

ensemble_model.compile(optimizer=Adam(lr=0.0045), loss='mse')
res = ensemble_model.fit_generator(train_gn, validation_data=val_gn, workers=1, epochs=3, callbacks=[check])

with open('/home/emariott/deepmagic/output_data/loss_history', 'wb') as f:
    pickle.dump(res, f)

# result, y_pred_test = snapshot_training(ensemble_model, train_gn, val_gn, test_gn, net_name,
#                                         max_lr=0.00045,
#                                         epochs=4,
#                                         snapshot_number=4
#                                         )
