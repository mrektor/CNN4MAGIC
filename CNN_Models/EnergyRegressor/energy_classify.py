from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from CNN_Models.EnergyRegressor.magic_inception import magic_inception
# from CNN_Models.EnergyRegressor.magic_inception import magic_inception
# from magic_inception import magic_inception
from utils import *

# %
x_train, y_train, x_test, y_test, input_shape = load_magic_data(logx=False, energy_th=2)
print(input_shape)

num_classes = 9

values_bin_train, bins = bin_data(y_train, num_bins=num_classes)
values_bin_test, _ = bin_data(y_test, num_bins=num_classes, bins=bins)

encoder = LabelEncoder()
y_train_cat = encoder.fit_transform(values_bin_train)
y_test_cat = encoder.transform(values_bin_test)

y_train_keras_cat = to_categorical(y_train_cat, num_classes=num_classes)
y_test_keras_cat = to_categorical(y_test_cat, num_classes=num_classes)
# %

# model_class = SEInceptionV3(include_top=True,
#                   weights=None,
#                   input_tensor=None,
#                   input_shape=input_shape,
#                   pooling=None,
#                   classes=num_classes)
#
# model_class.compile(loss=keras.losses.binary_crossentropy,
#                     optimizer=keras.optimizers.Adadelta(),
#                     metrics=['accuracy'])
model_class = magic_inception(input_shape=input_shape, num_filters_first_conv=132, dropout=0, num_classes=num_classes)
# model_class = deep_magic(input_shape, 'relu', num_classes)

model_class.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model_class.summary()
# %

batch_size = 512
epochs = 100

net_name = 'energy_class_inc_CBAMblock_sigmLeakyGlobalAvg_energyTh2_' + str(num_classes) + 'class'
tensorboard = TensorBoard(log_dir='energy_class')
early_stop = EarlyStopping(patience=8, min_delta=0.0001)
nan_stop = TerminateOnNaN()
check = ModelCheckpoint('/data/mariotti_data/checkpoints/grid_inc_filt_' + net_name + '.hdf5', period=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=4, min_lr=0.000005)

result = model_class.fit(x_train, y_train_keras_cat,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(x_test, y_test_keras_cat),
                         callbacks=[tensorboard, early_stop, nan_stop, reduce_lr, check])

# %%
y_pred_hot = model_class.predict(x_test)

num_bins = bins.shape[0]
bins_mean_value = np.zeros(num_bins - 1)
for i in range(num_bins - 1):
    bins_mean_value[i] = np.mean([bins[i], bins[i + 1]])

# %%
cm = confusion_matrix(y_true=np.argmax(y_test_keras_cat, 1), y_pred=np.argmax(y_pred_hot, 1))
plt.figure()
plot_confusion_matrix(cm=cm, classes=np.around(bins_mean_value, 2), normalize=True)
plt.savefig('/data/mariotti_data/pics/confusion_matrix_' + net_name + '.jpg')

# %%
# %
y_pred = encoder.inverse_transform(np.argmax(y_pred_hot, 1))
# plot_gaussian_error(y_test, y_pred, net_name=net_name + '_20bin', num_bins=20)
plot_gaussian_error(y_test, y_pred, net_name=net_name + '_' + str(num_classes) + 'bin', num_bins=num_classes)

# %
plot_hist2D(y_test, y_pred, net_name=net_name, num_bins=50)
