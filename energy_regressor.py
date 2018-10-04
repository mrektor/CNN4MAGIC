# %%
from __future__ import print_function

import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import load_model

# %% Data Loading
print('loading data')
with open('pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
    x_train = pickle.load(f)

with open('pickle_data/energy_train.pkl', 'rb') as f:
    raw_energy_train = pickle.load(f)

y_train = np.log(raw_energy_train)

with open('pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
    x_test = pickle.load(f)

with open('pickle_data/energy_test.pkl', 'rb') as f:
    raw_energy_test = pickle.load(f)

y_test = np.log(raw_energy_test)

print('Data dimensions:')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %
batch_size = 350

# input image dimensions
img_rows, img_cols = 67, 68

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# %%
# %
#
num_classes = 1
energy_regressor_net = Sequential()
energy_regressor_net.add(Conv2D(32, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=input_shape))
energy_regressor_net.add(Conv2D(32, (1, 1), activation='relu'))

energy_regressor_net.add(Conv2D(64, (3, 3), activation='relu'))
energy_regressor_net.add(Conv2D(64, (1, 1), activation='relu'))
energy_regressor_net.add(MaxPooling2D(pool_size=(2, 2)))

energy_regressor_net.add(Conv2D(128, (3, 3), activation='relu'))
energy_regressor_net.add(Conv2D(64, (1, 1), activation='relu'))
energy_regressor_net.add(MaxPooling2D(pool_size=(2, 2)))

energy_regressor_net.add(Conv2D(128, (3, 3), activation='relu'))
energy_regressor_net.add(Conv2D(64, (1, 1), activation='relu'))

energy_regressor_net.add(Conv2D(128, (3, 3), activation='relu'))
energy_regressor_net.add(Conv2D(64, (1, 1), activation='relu'))

energy_regressor_net.add(Conv2D(256, (3, 3), activation='relu'))

energy_regressor_net.add(GlobalAveragePooling2D())

# energy_regressor_net.add(Flatten())
# energy_regressor_net.add(Dense(100, activation='relu'))
# energy_regressor_net.add(Dropout(0.5))
# energy_regressor_net.add(Dense(30, activation='relu'))

energy_regressor_net.add(Dense(num_classes, activation='linear'))

energy_regressor_net.summary()

# %% Compile and fit
energy_regressor_net = load_model('checkpoints/energy_regressor_deep.hdf5')
energy_regressor_net.compile(loss=keras.losses.mean_squared_error,
                             optimizer=keras.optimizers.SGD(lr=0.0001))

tensorboard = TensorBoard(log_dir='logs_reg_2/')
early_stop = EarlyStopping(patience=3)
check = ModelCheckpoint('checkpoints/energy_regressor_deep.hdf5')

epochs = 20

energy_regressor_net.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=[tensorboard, early_stop, check])

# %%


# %% accuracy CURVE
y_pred = energy_regressor_net.predict(x_test)
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('true Energy (log)')
plt.ylabel('Energy prediction (log)')
plt.title('Regression Accuracy (Energy is in Log scale)')
plt.savefig('regression_accuracy_2.jpg')

score = energy_regressor_net.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score)
# print('Test accuracy:', score[1])
