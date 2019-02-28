from __future__ import print_function

from CNN4MAGIC.CNN_Models.BigData.clr import LRFinder
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_separation

# %%
BATCH_SIZE = 128
nb_epoch = 1  # Only finding lr
machine = '24cores'

# train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
#                                                                   want_golden=True,
#                                                                   want_position=True,
#                                                                   folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse_6_3punto5',
#                                                                   folder_point='/ssdraptor/magic_data/data_processed/point_like')

train_gn, val_gn = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=False,
    want_label=True,
    machine=machine,
    clean=True,
    include_time=False)

num_samples = len(val_gn)*BATCH_SIZE
# Exponential lr finder
# USE THIS FOR A LARGE RANGE SEARCH
# Uncomment the validation_data flag to reduce speed but get a better idea of the learning rate
lr_finder = LRFinder(num_samples, BATCH_SIZE, minimum_lr=5e-4, maximum_lr=10,
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

# %%Load Model
net_name = 'MobileNetV2_separation_clean_notime_128'

print('Loading the Neural Network...')
model = MobileNetV2_separation(alpha=1.0, include_time=False)
# model.compile(optimizer='sgd', loss='mse')
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#%
result = model.fit_generator(generator=val_gn,
                             # validation_data=val_gn,
                             epochs=1,
                             verbose=1,
                             callbacks=[lr_finder],
                             use_multiprocessing=True,
                             workers=24
                             )

lr_finder.plot_schedule(clip_beginning=10, clip_endding=5, net_name=net_name)
