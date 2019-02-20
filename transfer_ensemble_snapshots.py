import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_position
from CNN4MAGIC.Generator.training_util import superconvergence_training
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from glob import glob
from tqdm import tqdm

# %
BATCH_SIZE = 128

# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                    want_golden=True,
                                                                    want_position=True,
                                                                    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
                                                                    folder_point='/ssdraptor/magic_data/data_processed/point_like')
# %

snap_folder = '/home/emariott/deepmagic/output_data/snapshots'
snapshots = glob(f'{snap_folder}/MobileNetV2_4dense_position_beast_2019-02-19_18-40-38-*.h5')
snapshots_sort = sorted(snapshots)

# %
print(len(snapshots_sort[1:-1]))
# %
freeze = True

input_img = Input(shape=(67, 68, 4), name='ensemble_input')
feature_layers = []
for idx, snap in enumerate(tqdm(snapshots_sort[1:-1])):
    model = MobileNetV2_4dense_position(input_img)
    for single_layer in model.layers:
        tmp_name = single_layer.name
        single_layer.name = single_layer.name + f'_snap_{idx}'
    if freeze:
        for single_layer in model.layers:
            single_layer.trainable = False
    model.load_weights(snap)
    feature_layers.append(model.layers[-2].output)

one_layer = concatenate(feature_layers)

out_ens = Dense(2, kernel_regularizer='l2')(one_layer)

ensemble_model = Model(input_img, out_ens)

# %%
ensemble_model.summary()

# %%
BATCH_SIZE = 64
net_name = 'MV2-4D-30E-l2-EnsLast9'
superconvergence_training(ensemble_model, train_gn, val_gn, test_gn, net_name,
                          batch_size=BATCH_SIZE,
                          max_lr=0.001,
                          epochs=4,
                          patience=4,
                          model_checkpoint=1)
