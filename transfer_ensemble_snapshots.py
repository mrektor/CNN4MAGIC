import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_position
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# %%
BATCH_SIZE = 128

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_golden=True,
                                                                  want_position=True,
                                                                  folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
                                                                  folder_point='/ssdraptor/magic_data/data_processed/point_like')
# %%
from glob import glob

snap_folder = '/home/emariott/deepmagic/output_data/snapshots'
snapshots = glob(f'{snap_folder}/MobileNetV2_4dense_position_beast_2019-02-19_18-40-38-*.h5')
snapshots_sort = sorted(snapshots)
# %%
print(snapshots_sort)
input_img = Input(shape=(67, 68, 4), name='ensemble_input')
feature_layers = []
for snap in snapshots:
    model = MobileNetV2_4dense_position()
    model.load_weights(snap)
    feature_layers.append(model.layers[-2].output)

one_layer = concatenate(feature_layers)
out_ens = Dense(1, kernel_regularizer='l2')(one_layer)

ensemble_model = Model(input_img, out_ens)

# %%
