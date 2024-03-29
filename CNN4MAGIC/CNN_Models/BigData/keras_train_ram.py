import gc
import os
import pickle

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.optimizers import SGD

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.CNN_Models.BigData.cyclical_lr import CyclicLR
from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.stereo_models import *
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

# %%
# LOAD DATA
m1_tr, m2_tr, energy_tr = load_data_append('train', prune=True, impact=False)
m1_val, m2_val, energy_val = load_data_append('val', prune=True, impact=False)

energy_tr = np.log10(energy_tr)
energy_val = np.log10(energy_val)
# %%
# LOAD and COMPILE model
net_name = 'single_DenseNet_25_3_doubleDense-noImpact'

net_name_to_load = 'single_DenseNet_25_3_doubleDense-noImpact'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints/' + net_name_to_load + '.hdf5'

if os.path.exists(path):
    print('Loading model ' + net_name_to_load + '...')
    energy_regressor = load_model(path)
else:
    energy_regressor = single_DenseNet_25_3_doubleDense()

# energy_regressor = single_DenseNet_25_3()
EPOCHS = 40
opt = SGD(lr=0.0005)
energy_regressor.compile(optimizer=opt, loss='mse')

energy_regressor.summary()
gc.collect()

# %%
# M = 5  # number of snapshots
# nb_epoch = T = 120  # number of epochs
# alpha_zero = 0.15  # initial learning rate
# model_prefix = 'Model_'

# callbacks = SnapshotCallbackBuilder(T, M, alpha_zero).get_callbacks(model_prefix=net_name)

# early_stop = EarlyStopping(patience=5, min_delta=0.0001)
# nan_stop = TerminateOnNaN()
ten_dir = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/tensorboard_dir' + net_name
tensorboard = TensorBoard(log_dir=ten_dir, histogram_freq=1,
                          write_graph=False, write_images=False)
check = ModelCheckpoint('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints/' + net_name + '.hdf5',
                        period=1,
                        save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.000005)
# [:,:,:,1].reshape((134997, 67, 68, 1))

# callbacks.append(tensorboard)

clr = CyclicLR(base_lr=0.00003, max_lr=0.006,
               step_size=1000, mode='triangular')
clr_1 = OneCycleLR(batch_size=64, max_lr=0.005, num_samples=130814, num_epochs=EPOCHS)

result = energy_regressor.fit({'m1': m1_tr, 'm2': m2_tr}, energy_tr,
                              batch_size=64,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=({'m1': m1_val, 'm2': m2_val}, energy_val),
                              callbacks=[clr_1, check])

# %% Free memory
print('Freeing memory from training and validation data')
del m1_tr, m2_tr, energy_tr, m1_val, m2_val, energy_val
gc.collect()

# %% Save and plot stuff

m1_te, m2_te, energy_te = load_data_append('test', prune=True, impact=False)
y_test = np.log10(energy_te)

print('Making Predictions...')
y_pred = energy_regressor.predict({'m1': m1_te, 'm2': m2_te})

# %%
print('Plotting stuff...')
plot_hist2D(y_test, y_pred, fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/', net_name=net_name,
            num_bins=100)

plot_gaussian_error(y_test, y_pred, net_name=net_name + '_13bin', num_bins=13,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/')

# summarize history for loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/hystories/' + net_name + '_history.png')
plt.savefig('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/hystories/' + net_name + '_history.eps')

plt.show()

# %%
# Free memory
print('Freeing memory from test data')
del m1_te, m2_te
gc.collect()

print('Saving History')
with open('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/hystories/' + net_name + '_history.pkl', 'wb') as f:
    pickle.dump(result, f, protocol=4)

print('All done, everything went fine.')
