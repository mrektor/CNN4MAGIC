import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import MobileNetV2_energy_doubleDense

num_cores = 24

GPU=False
CPU=True

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

print('Loading the Neural Network...')
model = MobileNetV2_energy_doubleDense()
model.compile(optimizer='sgd', loss='mse')
model.summary()
print('Model Loaded.')

#%%
x_synt = np.random.randn(1000, 67, 68, 4)
y_synt = np.random.randn(1000, 1)
model.fit(x_synt, y_synt, 1)

#%%
BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)


# %% Train
EPOCHS = 30

net_name = 'MobileNetV2_energy_doubleDense-900kTrain'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints/' + net_name + '.hdf5'
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=5e-3,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)
stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=False,
                             workers=16
                             )

# %%
print('Training done.')
# %% plot training
import matplotlib.pyplot as plt

fig_folder = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/'
# summarize history for loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss MSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(fig_folder + net_name + '_loss_MSE.png')
plt.show()

plt.plot(result.history['mean_absolute_error'])
plt.plot(result.history['val_mean_absolute_error'])
plt.title('model loss MAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(fig_folder + net_name + '_loss_MAE.png')
plt.show()

plt.plot(result.history['mean_absolute_percentage_error'])
plt.plot(result.history['val_mean_absolute_percentage_error'])
plt.title('model loss MAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(fig_folder + net_name + '_loss_MAPE.png')
plt.show()

print('plotting training done')
