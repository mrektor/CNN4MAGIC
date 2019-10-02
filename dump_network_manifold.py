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
from CNN4MAGIC.Generator.models import efficientNet_B0_separation, VGG19_separation, efficientNet_B1_separation, \
    efficientNet_B2_separation, efficientNet_B3_separation, efficientNet_B4_separation, VGG16_separation


def pickle_read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pickle_dump(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
model = VGG19_separation()
net_name = 'VGG19_separation'
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
model.load_weights(
    f'/data4T/CNN4MAGIC/results/MC_classification/experiments/{net_name}/computed_data/final_{net_name}.h5')
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
# %
model.summary()
# %
# dense_1
from keras import Model

network_embedding = Model(inputs=model.input,
                          outputs=model.get_layer('global_max_pooling2d_2').output)

# %
crab_embedding = network_embedding.predict_generator(crab_generator, verbose=1)
# %
pickle_dump(
    f'/data4T/CNN4MAGIC/results/MC_classification/experiments/{net_name}/computed_data/crab_embedding.pkl',
    crab_embedding)

# %
crab_gammaness = pickle_read(
    f'/data4T/CNN4MAGIC/results/MC_classification/experiments/{net_name}/computed_data/crab_separation_{net_name}.pkl')
# %
print(crab_embedding.shape, crab_gammaness.shape)
# %
print(crab_embedding[0], crab_gammaness[0])


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

sc = RobustScaler()
pca = PCA(2)
projected_embedding = pca.fit_transform(sc.fit_transform(crab_embedding))

# %
print(pca.explained_variance_ratio_)
# %
plt.figure(figsize=(15, 15))
plt.scatter(projected_embedding[:, 0], projected_embedding[:, 1], c=crab_gammaness[:, 0])
plt.title(f'PCA 2D embedding (from {crab_embedding.shape[1]})')
plt.tight_layout()
plt.savefig(f'/data4T/CNN4MAGIC/results/MC_classification/experiments/{net_name}/plots/pca_2d_crab_embedding.png')
plt.close()

# %%
from sklearn.cluster import DBSCAN
from time import time
from tqdm import tqdm

eps_to_try = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 4, 6]

db_list = []
time_elapsed = []
for epss in tqdm(eps_to_try):
    bef = time()
    scanner = DBSCAN(n_jobs=8, eps=epss).fit(crab_embedding)
    now = time()
    time_elapsed.append(now-bef)
    db_list.append(scanner)

labelss = [db_single.labels_ for db_single in db_list]

# Number of clusters in labels, ignoring noise if present.
n_clusters_tot = []
n_noise_tot = []
for eps, labels in zip(eps_to_try, labelss):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    n_clusters_tot.append(n_clusters_)
    n_noise_tot.append(n_noise_)
    print(f'DB with eps: {eps}')
    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    print('')
# %%
print(crab_embedding.shape)
# %%
print(labels)

#%%
scanner = DBSCAN(n_jobs=8, eps=20).fit(crab_embedding)
#%%
labels = scanner.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
n_clusters_tot.append(n_clusters_)
n_noise_tot.append(n_noise_)
print(f'DB with eps: {eps}')
print(f'Estimated number of clusters: {n_clusters_}')
print(f'Estimated number of noise points: {n_noise_}')
print('')
# %%
# outliers = projected_embedding[projected_embedding[:, 1] > 5]
# print(outliers.shape)
# #%%
# outliers_bool = projected_embedding[:, 1] > 5
# idx_misclassified = np.where(outliers_bool)[0]
#
# batch_numbers = np.floor(idx_misclassified / BATCH_SIZE)
# idx_in_batches = np.mod(idx_misclassified, BATCH_SIZE)
#
# misclassified_events = [crab_generator[int(batch_number)][0][idx_in_batch] for batch_number, idx_in_batch in
#                         zip(batch_numbers, idx_in_batches)]
# #%%
# print(np.array(misclassified_events).shape)
# #%%
# folder_misc_complete = '/data4T/CNN4MAGIC/results/MC_classification/experiments/efficientNet_B3_last3_lin/plots/bizzarre_crab_events'
# for misclassified_number, single_event in enumerate(tqdm(misclassified_events)):
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#     i = 0
#     for ax in axes:
#         ax[0].imshow(single_event[:, :, i])
#         ax[1].imshow(single_event[:, :, i + 1])
#         i += 2
#     plt.suptitle(f'Bizzarre event {misclassified_number}')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f'{folder_misc_complete}/event_{misclassified_number}.png')
#     plt.close()
