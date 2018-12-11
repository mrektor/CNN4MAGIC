from __future__ import print_function

import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from CNN4MAGIC.CNN_Models.BigData.clr import LRFinder
from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.stereo_models import *

if not os.path.exists('weights/'):
    os.makedirs('weights/')

net_name = 'single-SE-DenseNet-10-5-Gold'

weights_file = 'weights/' + net_name + '.h5'
model_checkpoint = ModelCheckpoint(weights_file, save_best_only=True,
                                   save_weights_only=True)

batch_size = 64
nb_epoch = 1  # Only finding lr
data_augmentation = False

# The data, shuffled and split between train and test sets:
m1_tr, m2_tr, energy_tr = load_data_append('train', prune=True)
m1_val, m2_val, energy_val = load_data_append('val', prune=True)

energy_tr = np.log10(energy_tr)
energy_val = np.log10(energy_val)
# Learning rate finder callback setup
num_samples = m1_tr.shape[0]

# Exponential lr finder
# USE THIS FOR A LARGE RANGE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
lr_finder = LRFinder(num_samples, batch_size, minimum_lr=1e-4, maximum_lr=20,
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
LRFinder.plot_schedule_from_file('weights/', clip_beginning=10, clip_endding=5)

# For training, the auxilary branch must be used to correctly train NASNet

model = single_DenseNet_10_5()
model.summary()

optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=optimizer)

# model.load_weights(weights_file)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit({'m1': m1_tr, 'm2': m2_tr}, energy_tr,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=({'m1': m1_val, 'm2': m2_val}, energy_val),
              shuffle=True,
              verbose=1,
              callbacks=[lr_finder, model_checkpoint])

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5, net_name=net_name)
