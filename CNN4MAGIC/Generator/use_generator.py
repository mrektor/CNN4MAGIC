from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import *

BATCH_SIZE = 400
train_gn, val_gn, position = load_data_generators(batch_size=BATCH_SIZE, want_position=True)


# %% Load Model
# print('Loading the Neural Network...')
# model = NASNet_mobile_position()
# model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
# model.summary()
#
# # %% Train
# EPOCHS = 20
#
# net_name = 'NASNET-mobile-position'
# path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name
# check = ModelCheckpoint(filepath=path, save_best_only=True)
# clr = OneCycleLR(max_lr=0.01,
#                  num_epochs=EPOCHS,
#                  num_samples=len(train_gn),
#                  batch_size=BATCH_SIZE)
# stop = EarlyStopping(patience=2)
#
# result = model.fit_generator(generator=train_gn,
#                              validation_data=val_gn,
#                              epochs=EPOCHS,
#                              verbose=1,
#                              callbacks=[check, clr, stop],
#                              use_multiprocessing=True,
#                              workers=16
#                              )
# %%
# print('Loading the Neural Network...')
# model = DenseNet121_position()
# model.compile(optimizer='sgd', loss='mse')
# model.summary()
#
# # %% Train
# EPOCHS = 30
#
# net_name = 'DenseNet121-mobile-position'
# path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name + '.hdf5'
# check = ModelCheckpoint(filepath=path, save_best_only=True, period=5)
# clr = OneCycleLR(max_lr=1e-2,
#                  num_epochs=EPOCHS,
#                  num_samples=len(train_gn),
#                  batch_size=BATCH_SIZE)
# stop = EarlyStopping(patience=2)
#
# result = model.fit_generator(generator=train_gn,
#                              validation_data=val_gn,
#                              epochs=EPOCHS,
#                              verbose=1,
#                              callbacks=[check, clr, stop],
#                              use_multiprocessing=True,
#                              workers=16,
#                              max_queue_size=100
#                              )
# ######
# #%%
# print('Loading the Neural Network...')
# model = MobileNetV2_slim_position()
# model.compile(optimizer='sgd', loss='mse')
# model.summary()
#
# # %% Train
# EPOCHS = 30
#
# net_name = 'MobileNetV2-slim-position'
# path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name +'.hdf5'
# check = ModelCheckpoint(filepath=path, save_best_only=True)
# clr = OneCycleLR(max_lr=1e-4,
#                  num_epochs=EPOCHS,
#                  num_samples=len(train_gn),
#                  batch_size=BATCH_SIZE)
# stop = EarlyStopping(patience=2)
#
# result = model.fit_generator(generator=train_gn,
#                              validation_data=val_gn,
#                              epochs=EPOCHS,
#                              verbose=1,
#                              callbacks=[check, clr, stop],
#                              use_multiprocessing=True,
#                              workers=16,
#                              max_queue_size=100
#                              )
#######
# %%
print('Loading the Neural Network...')
model = MobileNetV2_position()
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
model.summary()

# %% Train
EPOCHS = 30

net_name = 'MobileNetV2-alpha1-position'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=1e-4,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)
stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=True,
                             workers=8,
                             max_queue_size=5
                             )