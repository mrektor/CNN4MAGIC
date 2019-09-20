import matplotlib

matplotlib.use('TkAgg')
# from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
# from CNN4MAGIC.Generator.models import dummy_cnn
# from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 256
machine = 'towerino'

# % Load the data
import pickle


def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


# % Load complement
folder_complement = '/ssdraptor/magic_data/classification_MC/complementary'
diffuse_filenames_list, diffuse_labels, diffuse_energies, diffuse_positions = pickle_read(
    f'{folder_complement}/diffuse_complement.pkl')
point_filenames_list, point_labels, point_energies, point_positions = pickle_read(
    f'{folder_complement}/point_complement.pkl')
diffuse_df = pickle_read(f'{folder_complement}/diffuse_df.pkl')
point_df = pickle_read(f'{folder_complement}/point_df.pkl')
protons_complement = pickle_read(f'{folder_complement}/protons_big_df_ID_Labels.pkl')

protons_labels = protons_complement.set_index('ID').to_dict()['label']

# % Point to all files folder
npy_folder = '/ssdraptor/magic_data/classification_MC/all_npy'

# % Make Clean
golden_df_diffuse = diffuse_df[
    (diffuse_df['intensity_M1'] > 50) &
    (diffuse_df['intensity_M2'] > 50) &
    (diffuse_df['leakage2_pixel_M1'] < 0.2) &
    (diffuse_df['leakage2_pixel_M2'] < 0.2)
    ]

golden_df_point = point_df[
    (point_df['intensity_M1'] > 50) &
    (point_df['intensity_M2'] > 50) &
    (point_df['leakage2_pixel_M1'] < 0.2) &
    (point_df['leakage2_pixel_M2'] < 0.2)
    ]

want_golden = True
if want_golden:
    ids_diffuse = golden_df_diffuse['ID'].values
    ids_point = golden_df_point['ID'].values
else:
    ids_diffuse = diffuse_df['ID'].values
    ids_point = point_df['ID'].values

ids_protons = protons_complement['ID'].values
# % tr-va-te split

# num_protons = len(ids_protons)
# ids_protons_tr = ids_protons[:int(num_protons * 0.7)]
# ids_protons_va = ids_protons[int(num_protons * 0.7):int(num_protons * 0.8)]
# ids_protons_test = ids_protons[int(num_protons * 0.8):]

num_diffuse = len(ids_diffuse)
ids_diffuse_tr = ids_diffuse[:int(num_diffuse * 0.75)]
ids_diffuse_va = ids_diffuse[int(num_diffuse * 0.75):int(num_diffuse * 0.85)]
ids_diffuse_te = ids_diffuse[int(num_diffuse * 0.85):]

# num_point = len(ids_point)
# ids_point_tr = ids_point[:int(num_point * 0.6)]
# ids_point_va = ids_point[:int(num_point * 0.2)]
# ids_point_te = ids_point[int(num_point * 0.2):int(num_point * 0.4)]

# % Define file list
# train_list_global = list(ids_protons_tr) + list(ids_diffuse_tr)
# validation_list_global = list(ids_protons_va) + list(ids_diffuse_va) + list(ids_point_va)
# test_list_global = list(ids_protons_test) + list(ids_diffuse_te) + list(ids_point_te)

# %
# print(len(ids_protons_tr), len(ids_diffuse_tr))
# print(len(ids_protons_va), len(ids_diffuse_va), len(ids_point_va))
# print(len(ids_protons_test), len(ids_diffuse_te), len(ids_point_te))
# % Define global label lookup dictionary
# global_lookup_labels = dict()
# global_lookup_labels.update(point_labels)
# global_lookup_labels.update(diffuse_labels)
# global_lookup_labels.update(protons_labels)
# print(len(global_lookup_labels))
# % shuffle (might not be necessary)
import random

random.seed(42)
random.shuffle(ids_diffuse_tr)

# % define generators
from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator

# %
train_gn = MAGIC_Generator(list_IDs=list(ids_diffuse_tr),
                           labels=diffuse_positions,
                           position=True,
                           batch_size=BATCH_SIZE,
                           folder=npy_folder,
                           apply_log_to_raw=False,
                           include_time=True
                           )

val_gn = MAGIC_Generator(list_IDs=list(ids_diffuse_va),
                         labels=diffuse_positions,
                         position=True,
                         shuffle=False,
                         batch_size=BATCH_SIZE,
                         folder=npy_folder,
                         apply_log_to_raw=False,
                         include_time=True
                         )
# %%
from keras_radam import RAdam
# from CNN4MAGIC.Generator.models import MobileNetV2_separation
# from CNN4MAGIC.Other_utilities.resnext import ResNext
from CNN4MAGIC.Other_utilities.keras_efficientnets.efficientnet import EfficientNetB0
from keras.layers import Input, Dense, Flatten
from keras.models import Model

input_img = Input(shape=(67, 68, 4), name='m1m2')
body = EfficientNetB0(input_tensor=input_img, include_top=False, weights=None)
x = body.layers[-1].output
x = Flatten()(x)
head = Dense(2)(x)
model = Model(inputs=input_img, output=head)

model.compile(optimizer=RAdam(), loss='mse')
model.summary()
# %
from keras.callbacks import ModelCheckpoint
result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=12,
                             verbose=1,
                             callbacks=[ModelCheckpoint('/ssdraptor/magic_data/position_coord/eff-b0.hdf5')],
                             use_multiprocessing=False,
                             workers=8)
pickle_dump('/ssdraptor/magic_data/position_coord/history_callback.pkl', result)

