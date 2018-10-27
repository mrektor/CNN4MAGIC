# %%
from __future__ import print_function

from CNN_Models.EnergyRegressor.magic_inception import magic_inception
from utils import *

# % Data Loading
x_train, y_train, x_test, y_test, input_shape = load_magic_data(logx=False, energy_th=0)

model_reg = magic_inception(input_shape=input_shape, num_filters_first_conv=132, dropout=0, num_classes=1)

model_reg.compile(optimizer='adam', loss='mse')

net_name = 'energy_class_reg_magicInception_CBAM_th2'

early_stop = EarlyStopping(patience=8, min_delta=0.0001)
nan_stop = TerminateOnNaN()
check = ModelCheckpoint('/data/mariotti_data/checkpoints/' + net_name + '.hdf5', period=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                              patience=4, min_lr=0.000005)

result = model_reg.fit(x_train, y_train,
                       batch_size=450,
                       epochs=100,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=[early_stop, nan_stop, reduce_lr, check])

# %%


y_pred = model_reg.predict(x_test)

plot_hist2D(y_test, y_pred, net_name=net_name, num_bins=50)

plot_gaussian_error(y_test, y_pred, net_name=net_name + '_10bin', num_bins=10)
# plot_gaussian_error(y_test, y_pred, net_name=net_name + '_20bin', num_bins=20)

# %
