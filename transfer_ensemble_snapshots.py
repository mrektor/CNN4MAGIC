import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SE_InceptionV3_SingleDense_energy
from keras.layers import Input, Dense, concatenate, BatchNormalization, LeakyReLU
from keras.models import Model
from glob import glob
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

# %%
BATCH_SIZE = 32
machine = 'towerino'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    clean=False)
# %

snap_folder = '/home/emariott/deepmagic/output_data/snapshots'
# %%
snapshots = glob(f'{snap_folder}/SE_InceptionV3_SingleDense_energy_yesTime_from40*.h5')
snapshots_sort = sorted(snapshots)

# %
print(snapshots_sort)
print(len(snapshots_sort))

# %%
snap_to_ensemble = [snapshots_sort[1]] + snapshots_sort[-6:-1]
print(snap_to_ensemble)
print(len(snap_to_ensemble))
# %%
freeze = False

input_img = Input(shape=(67, 68, 4), name='ensemble_input')
feature_layers = []
for idx, snap in enumerate(tqdm(snap_to_ensemble)):
    model = SE_InceptionV3_SingleDense_energy(True, input_img)
    for single_layer in model.layers:
        tmp_name = single_layer.name
        single_layer.name = single_layer.name + f'_snap_{idx}'
    if freeze:
        for single_layer in model.layers:
            single_layer.trainable = False
    model.load_weights(snap)
    feature_layers.append(model.layers[-2].output)

one_layer = concatenate(feature_layers)
out_ens = BatchNormalization()(one_layer)

out_ens = Dense(64)(one_layer)
out_ens = BatchNormalization()(out_ens)
out_ens = LeakyReLU()(out_ens)

out_ens = Dense(1, kernel_regularizer='l2')(out_ens)

ensemble_model = Model(input_img, out_ens)

# %%
ensemble_model.summary()

# %%
net_name = 'Tranfer_Ensemble_SE_InceptionV3_SingleDense_energy_from40_last6_nofreeze_dense64_adam4e-4'
check = ModelCheckpoint(f'output_data/checkpoints/{net_name}.hdf5', save_best_only=True)

ensemble_model.compile(optimizer=Adam(lr=4e-4), loss='mse')
res = ensemble_model.fit_generator(train_gn, validation_data=val_gn, workers=8, epochs=5, callbacks=[check])

with open('/home/emariott/deepmagic/output_data/loss_history', 'wb') as f:
    pickle.dump(res, f)

# result, y_pred_test = snapshot_training(ensemble_model, train_gn, val_gn, test_gn, net_name,
#                                         max_lr=0.00045,
#                                         epochs=4,
#                                         snapshot_number=4
#                                         )
