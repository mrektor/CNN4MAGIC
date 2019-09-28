import matplotlib

matplotlib.use('agg')

import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
# from CNN4MAGIC.Generator.training_util import get_telegram_callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_radam import RAdam
from tqdm import tqdm

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import efficientNet_B0_separation, efficientNet_B1_separation, \
    efficientNet_B2_separation, efficientNet_B3_separation, efficientNet_B4_separation


def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
model = efficientNet_B3_separation(dropout=0, drop_connect=0.5, last_is_three=True, nonlinear_last=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
model.load_weights(
    '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/computed_data/final_efficientNet_B3_last3_lin.h5')
# %%
BATCH_SIZE = 128
big_df_crab, crab_evt_list = pickle_read(
    '/ssdraptor/magic_data/crab/crab_data/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl')
labels_crab = {ID: 0 for ID in crab_evt_list}  # Dummy
crab_generator = MAGIC_Generator(list_IDs=crab_evt_list,
                                 labels=labels_crab,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/ssdraptor/magic_data/crab/crab_data/crab_npy',
                                 include_time=True)
# %%
model.summary()
# %%
# dense_1
from keras import Model

network_embedding = Model(inputs=model.input,
                          outputs=model.get_layer('dense_3').output)

# %%
crab_embedding = network_embedding.predict_generator(crab_generator, verbose=1)
# %
pickle_dump(
    '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/computed_data/crab_embedding.pkl',
    crab_embedding)

# %%
crab_gammaness = pickle_read(
    '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/computed_data/crab_separation_efficientNet_B3_last3_lin.pkl')
# %%
print(crab_embedding.shape, crab_gammaness.shape)
# %%
print(crab_embedding[0], crab_gammaness[0])

# %%
fig = plt.figure(figsize=(20, 20))
ax = plt.axes(projection="3d")

ax.scatter3D(crab_embedding[:, 0], crab_embedding[:, 1], crab_embedding[:, 2], c=crab_gammaness, marker='.')
plt.savefig(
    '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/plots/crab_embedding.png')
plt.close()
