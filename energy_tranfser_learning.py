from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.models import load_model, Model

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 400
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

# %%
print('Loading the Neural Network...')

net_name = 'MobileNetV2-alpha1-position-buonanno'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name

print('Loading model ' + net_name + '...')
model = load_model(path)

# %%
global_avg_pool = model.layers[-2].output
out = Dense(1, name='energy')(global_avg_pool)
new_model = Model(model.input, out)
new_model.compile(optimizer='sgd', loss='mse')
new_model.summary()
# %% Train
EPOCHS = 30

net_name = 'MobileNetV2-alpha1-buonanno-energy-transfer'
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
                             workers=1,
                             max_queue_size=30
                             )

# %%
import gc

gc.collect()

from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn)
y_test = energy_te
print('Plotting stuff...')
plot_hist2D(y_test, y_pred,
            fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/',
            net_name=net_name,
            num_bins=100)

plot_gaussian_error(y_test, y_pred,
                    net_name=net_name + '_13bin',
                    num_bins=13,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/')
print('plotting metrics done')

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
