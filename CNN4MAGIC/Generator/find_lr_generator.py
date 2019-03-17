from __future__ import print_function

from CNN4MAGIC.CNN_Models.BigData.clr import LRFinder
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SE_incres_energy

# %
BATCH_SIZE = 128
nb_epoch = 1  # Only finding lr
machine = 'towerino'

# train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
#                                                                   want_golden=True,
#                                                                   want_position=True,
#                                                                   folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse_6_3punto5',
#                                                                   folder_point='/ssdraptor/magic_data/data_processed/point_like')

train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    want_log_energy=True,
    # apply_log_to_raw=False,
    machine=machine,
    clean=False,
    include_time=True)

# %%
num_samples = len(val_gn)*BATCH_SIZE
# Exponential lr finder
# USE THIS FOR A LARGE RANGE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
lr_finder = LRFinder(num_samples, BATCH_SIZE, minimum_lr=0.0001, maximum_lr=50,
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

# %Load Model
print('Loading the Neural Network...')

model = SE_incres_energy()
net_name = 'SE_incre_energy'
# model = energy_skrr(False)
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/energy_skrr_fromEpoch30_2019-03-13_03-14-22-15.h5')
# %
# net_name = 'energy_skrr_30_best'

# model.compile(optimizer='sgd', loss='mse')
model.compile(optimizer='sgd', loss='mse')
model.summary()

import matplotlib.pyplot as plt

plt.figure()
#%
result = model.fit_generator(generator=val_gn,
                             # validation_data=val_gn,
                             epochs=1,
                             verbose=1,
                             callbacks=[lr_finder],
                             use_multiprocessing=False,
                             workers=8
                             )

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5, net_name=net_name)

# %%
