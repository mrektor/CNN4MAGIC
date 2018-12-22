from __future__ import print_function

import glob
import os
import random

from tqdm import tqdm

from CNN4MAGIC.CNN_Models.BigData.clr import LRFinder
from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import *

if not os.path.exists('weights/'):
    os.makedirs('weights/')
import pickle as pkl

net_name = 'MobilenetV2Slim_position'
# weights_file = 'weights/' + net_name + '.h5'
# model_checkpoint = ModelCheckpoint(weights_file, save_best_only=True,
#                                    save_weights_only=True)

batch_size = 64
nb_epoch = 1  # Only finding lr
data_augmentation = False

# The data, shuffled and split between train and test sets:

# load IDs
print('Loading labels...')
filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total.pkl'
with open(filename, 'rb') as f:
    data, energy, labels, position = pkl.load(f)

# %%
import numpy as np

energy = {k: np.log10(v) for k, v in energy.items()}  # Convert energies in log10

# %% puto coñazo da cambiare pronto
eventList_total = glob.glob('/data2T/mariotti_data_2/MC_npy/partial_dump_MC/*')
newlist = []
for event in tqdm(eventList_total):
    newlist.append(event[47:-4])

eventList_total = newlist
random.seed(42)
random.shuffle(eventList_total)
num_files = len(eventList_total)
partition = {}
partition['train'] = eventList_total[:int(num_files / 2)]
partition['validation'] = eventList_total[int(num_files / 2):int(num_files * 3 / 2)]
partition['test'] = eventList_total[int(num_files * 3 / 2):]


def clean_missing_data(data, labels):
    p = 0
    todelete = []
    for key in data:
        try:
            a = labels[key]
        except KeyError:
            todelete.append(key)
            p = p + 1
    print(f'{len(todelete)} of KeyErrors')
    for key in todelete:
        data.remove(key)
    return data


partition['train'] = clean_missing_data(partition['train'], position)
partition['test'] = clean_missing_data(partition['test'], position)
partition['validation'] = clean_missing_data(partition['validation'], position)

num_samples = len(partition['train'])

# Exponential lr finder
# USE THIS FOR A LARGE RANGE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
lr_finder = LRFinder(num_samples, batch_size, minimum_lr=1e-6, maximum_lr=20,
                     lr_scale='exp',
                     # validation_data=({'m1': m1_val, 'm2': m2_val}, energy_val),  # use the validation data for losses
                     validation_sample_rate=5,
                     save_dir='weights/', verbose=True)

# Linear lr finder
# USE THIS FOR A CLOSE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
# lr_finder = LRFinder(num_samples, batch_size, minimum_lr=5e-4, maximum_lr=1e-2,
#                      lr_scale='linear',
#                      validation_data=(X_test, y_test),  # use the validation data for losses
#                      validation_sample_rate=5,
#                      save_dir='weights/', verbose=True)

# plot the previous values if present
# LRFinder.plot_schedule_from_file('weights/', clip_beginning=10, clip_endding=5)

# For training, the auxilary branch must be used to correctly train NASNet

BATCH_SIZE = 64
train_gn = MAGIC_Generator(list_IDs=partition['train'],
                           labels=position,
                           position=True,
                           batch_size=BATCH_SIZE,
                           folder='/data2T/mariotti_data_2/MC_npy/partial_dump_MC'
                           )

# %% Load Model
print('Loading the Neural Network...')
model = MobileNetV2_slim_position()
model.compile(optimizer='sgd', loss='mse')
model.summary()
# model.load_weights(weights_file)


result = model.fit_generator(generator=train_gn,
                             # validation_data=val_gn,
                             epochs=1,
                             verbose=1,
                             callbacks=[lr_finder],
                             use_multiprocessing=True,
                             workers=16
                             )

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5, net_name=net_name)
