import gc
import glob
import pickle
import random
import time

import keras
import numpy as np


# %%
class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32), n_channels=2,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


def load_data_train(pruned=False):
    fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last/*.pkl')
    if pruned:
        fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last_pruned/*.pkl')

    random.seed(42)
    random.shuffle(fileList)
    # %%
    befbef = time.time()
    times = []
    tot_items = 134997
    if pruned:
        tot_items = 130814
    full_energy = np.zeros(tot_items)
    full_interp_M1 = np.zeros((tot_items, 67, 68, 2))
    full_interp_M2 = np.zeros((tot_items, 67, 68, 2))

    old = 0
    print(f'number of files: {len(fileList)}')
    print('start loading...')
    for i, file in enumerate(fileList[:1500]):
        if i % 5 == 0:
            print('Loading training data: ' + str(int(i * 10000 / len(fileList[:1500])) / 100) + '%')
        bef = time.time()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            num_items = len(data['energy'])
            full_energy[old:old + num_items] = data['energy']
            full_interp_M1[old:old + num_items, :, :, :] = data['M1_interp']
            full_interp_M2[old:old + num_items, :, :, :] = data['M2_interp']
            old = old + num_items
        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    print('Number of items: ' + str(len(full_energy)))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')

    return full_interp_M1, full_interp_M2, full_energy


def load_data_val(pruned=False):
    fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last/*.pkl')
    if pruned:
        fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last_pruned/*.pkl')

    random.seed(42)
    random.shuffle(fileList)
    # %%
    befbef = time.time()
    times = []
    tot_items = 67322
    if pruned:
        tot_items = 65194
    full_energy = np.zeros(tot_items)
    full_interp_M1 = np.zeros((tot_items, 67, 68, 2))
    full_interp_M2 = np.zeros((tot_items, 67, 68, 2))

    old = 0
    print(f'number of files: {len(fileList[1500:2250])}')
    print('start loading...')
    for i, file in enumerate(fileList[1500:2250]):
        if i % 5 == 0:
            print('Loading validation data: ' + str(int(i * 10000 / len(fileList[1500:2250])) / 100) + '%')
        bef = time.time()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            num_items = len(data['energy'])
            full_energy[old:old + num_items] = data['energy']
            full_interp_M1[old:old + num_items, :, :, :] = data['M1_interp']
            full_interp_M2[old:old + num_items, :, :, :] = data['M2_interp']
            old = old + num_items
        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    print('Number of items: ' + str(len(full_energy)))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')

    return full_interp_M1, full_interp_M2, full_energy


def load_data_test(pruned=False):
    fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last/*.pkl')
    if pruned:
        fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last_pruned/*.pkl')

    random.seed(42)
    random.shuffle(fileList)
    # %%
    befbef = time.time()
    times = []
    tot_items = 67509
    if pruned:
        tot_items = 65388
    full_energy = np.zeros(tot_items)
    full_interp_M1 = np.zeros((tot_items, 67, 68, 2))
    full_interp_M2 = np.zeros((tot_items, 67, 68, 2))

    old = 0
    print(f'number of files: {len(fileList[2250:])}')
    print('start loading...')
    for i, file in enumerate(fileList[2250:]):
        if i % 5 == 0:
            print('Loading test data: ' + str(int(i * 10000 / len(fileList[2250:])) / 100) + '%')
        bef = time.time()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            num_items = len(data['energy'])
            full_energy[old:old + num_items] = data['energy']
            full_interp_M1[old:old + num_items, :, :, :] = data['M1_interp']
            full_interp_M2[old:old + num_items, :, :, :] = data['M2_interp']
            old = old + num_items
        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    print('Number of items: ' + str(len(full_energy)))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')

    return full_interp_M1, full_interp_M2, full_energy


# %%


# fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC_channel_last_pruned/*.pkl')
# random.seed(42)
# random.shuffle(fileList)
# # %%
#
# befbef = time.time()
# times = []
# tot_items = 0
# full_energy = np.zeros(tot_items)
# # full_interp_M1 = np.zeros((tot_items, 67, 68, 2))
# # full_interp_M2 = np.zeros((tot_items, 67, 68, 2))
#
# old = 0
# print(f'number of files: {len(fileList)}')
# print('start loading...')
# for i, file in enumerate(fileList[2250:]):
#     if i % 5 == 0:
#         print('Loading training data: ' + str(int(i * 10000 / len(fileList[:1500])) / 100) + '%')
#     bef = time.time()
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         num_items = len(data['energy'])
#         tot_items += num_items
#         # full_energy[old:old + num_items] = data['energy']
#         # full_interp_M1[old:old + num_items, :, :, :] = data['M1_interp']
#         # full_interp_M2[old:old + num_items, :, :, :] = data['M2_interp']
#         old = old + num_items
#     now = time.time()
#     times.append(now - bef)
# nownow = time.time()
#
# print('Number of items: ' + str(len(full_energy)))
# print(f'Time for loading all the files: {nownow-befbef}')
# print(f'Average time for loading one dict: {np.mean(np.array(times))}')
#
# print(tot_items)
# %%
from tqdm import tqdm


def load_data_append(which='train', fileListFolder='/data2T/mariotti_data_2/interp_from_root/MC_channel_last_full',
                     prune=False):
    fileList = glob.glob(fileListFolder + '/*.pkl')

    if len(fileList) == 0:
        print('The directory does not contain any file to load')
        raise ValueError

    random.seed(42)
    random.shuffle(fileList)

    befbef = time.time()
    times = []

    full_energy = []
    full_interp_M1 = []
    full_interp_M2 = []

    if which == 'train':
        toLoad = fileList[:1500]
        print('Loading TRAIN data')

    if which == 'val':
        toLoad = fileList[1500:2000]
        print('Loading VALIDATION data')


    if which == 'test':
        toLoad = fileList[2000:]
        print('Loading TEST data')

    if which == 'debug':
        toLoad = fileList[:2]
        print('Loading DEBUG data')

    print(f'number of files: {len(toLoad)}')
    print('start loading...')
    for i, file in enumerate(tqdm(toLoad)):

        bef = time.time()
        with open(file, 'rb') as f:
            data = pickle.load(f)
            if prune:
                # Conditions
                impact = data['impact'] < 80000
                intensity_ok = data['extras1']['intensity'] > 100
                leak_ok = data['extras1']['leakage2_pixel'] < 0.2
                condition = np.logical_and(impact, intensity_ok)
                condition = np.logical_and(condition, leak_ok)

                # Pruning
                full_energy.append(data['energy'][condition].reshape(
                    (data['energy'][condition].shape[0], 1)))  # Add one axis for ease of vstack
                full_interp_M1.append(data['M1_interp'][condition])
                full_interp_M2.append(data['M2_interp'][condition])
            else:
                full_energy.append(
                    data['energy'].reshape((data['energy'].shape[0], 1)))  # Add one axis for ease of vstack
                full_interp_M1.append(data['M1_interp'])
                full_interp_M2.append(data['M2_interp'])
        now = time.time()
        times.append(now - bef)
    nownow = time.time()

    full_energy = np.vstack(full_energy).flatten()
    full_interp_M1 = np.vstack(full_interp_M1)
    gc.collect()

    full_interp_M2 = np.vstack(full_interp_M2)

    print('Number of items: ' + str(len(full_energy)))
    print(f'Time for loading all the files: {nownow-befbef}')
    print(f'Average time for loading one dict: {np.mean(np.array(times))}')
    print('cleaning memory...')
    gc.collect()
    print('cleaned.')


    return full_interp_M1, full_interp_M2, full_energy
