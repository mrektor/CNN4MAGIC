from __future__ import print_function

import os

from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import SGD

from CNN4MAGIC.CNN_Models.BigData.clr import LRFinder
from CNN4MAGIC.CNN_Models.SeparationStereo.stereo_separation_models import *
from CNN4MAGIC.CNN_Models.SeparationStereo.utils import load_separation_data

if not os.path.exists('weights/'):
    os.makedirs('weights/')

net_name = 'single_DenseNet_25_3_doubleDense'

weights_file = 'weights/' + net_name + '.h5'
model_checkpoint = ModelCheckpoint(weights_file, save_best_only=True,
                                   save_weights_only=True)

batch_size = 64
nb_epoch = 1  # Only finding lr
data_augmentation = False

# The data, shuffled and split between train and test sets:
m1_tr, m2_tr, label_tr = load_separation_data('train')
# m1_val, m2_val, energy_val = load_data_append('val', prune=True)

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

model = single_DenseNet_25_3_doubleDense()
model.summary()

optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

# model.load_weights(weights_file)

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit({'m1': m1_tr, 'm2': m2_tr}, label_tr,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              callbacks=[lr_finder, model_checkpoint])

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5, net_name=net_name)
