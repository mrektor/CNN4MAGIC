import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2
from keras.layers import Input, Dense, concatenate
from CNN4MAGIC.Generator.training_util import snapshot_training

from keras.models import Model
from glob import glob
from tqdm import tqdm

# %%
BATCH_SIZE = 32
machine = 'towerino'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    clean=False)
# %%

snap_folder = '/home/emariott/deepmagic/output_data/snapshots'

snapshots = glob(f'{snap_folder}/SEDenseNet121_position_noclean_Gold_2019-02-25_01-37-25-*.h5')
snapshots_sort = sorted(snapshots)

# %
print(snapshots_sort)
print(len(snapshots_sort))


# %%
snap_to_ensemble = [snapshots_sort[-5]] + [snapshots_sort[1]] + [snapshots_sort[3]]
print(snap_to_ensemble)
print(len(snap_to_ensemble))
# %%
freeze = True

input_img = Input(shape=(67, 68, 4), name='ensemble_input')
feature_layers = []
for idx, snap in enumerate(tqdm(snap_to_ensemble)):
    model = SEDenseNet121_position_l2(input_img, True)
    for single_layer in model.layers:
        tmp_name = single_layer.name
        single_layer.name = single_layer.name + f'_snap_{idx}'
    if freeze:
        for single_layer in model.layers:
            single_layer.trainable = False
    model.load_weights(snap)
    feature_layers.append(model.layers[-1].output)

one_layer = concatenate(feature_layers)
# out_ens = BatchNormalization()(one_layer)
# %%
# out_ens = Dropout(0.5)(out_ens)
# out_ens = Dense(256)(out_ens)
# out_ens = BatchNormalization()(out_ens)
# out_ens = ReLU()(out_ens)

# out_ens = Dropout(0.2)(out_ens)
# out_ens = Dense(64, kernel_regularizer='l2')(out_ens)
# out_ens = BatchNormalization()(out_ens)
# out_ens = LeakyReLU()(out_ens)

out_ens = Dense(2, kernel_regularizer='l2')(one_layer)

ensemble_model = Model(input_img, out_ens)

# %
ensemble_model.summary()

# %%
net_name = 'tranfer-SeDense121-position-bestSnaps_6_10_12_LR5e-3_Single256Dense_drop05'
# check = ModelCheckpoint(f'output_data/checkpoints/{net_name}.hdf5', save_best_only=True)
#
# ensemble_model.compile(optimizer=Adam(lr=4e-4), loss='mse')
# res = ensemble_model.fit_generator(train_gn, validation_data=val_gn, workers=8, epochs=5, callbacks=[check])
#
# with open('/home/emariott/deepmagic/output_data/loss_history', 'wb') as f:
#     pickle.dump(res, f)

result, y_pred_test = snapshot_training(model=ensemble_model,
                                        train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                                        net_name=net_name,
                                        max_lr=1e-5,
                                        epochs=10,
                                        snapshot_number=10,
                                        swa=4
                                        )
