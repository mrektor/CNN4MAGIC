# %%
from __future__ import print_function

import pickle

from models import *
from resnet import *
from utils import *

# % Data Loading
print('loading data')
with open('pickle_data/gamma_energy_numpy_train.pkl', 'rb') as f:
    x_train = pickle.load(f)

with open('pickle_data/energy_train.pkl', 'rb') as f:
    raw_energy_train = pickle.load(f)

y_train = np.log10(raw_energy_train)

with open('pickle_data/gamma_energy_numpy_test.pkl', 'rb') as f:
    x_test = pickle.load(f)

with open('pickle_data/energy_test.pkl', 'rb') as f:
    raw_energy_test = pickle.load(f)

y_test = np.log10(raw_energy_test)

print('Data dimensions:')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %
batch_size = 256

# input image dimensions
img_rows, img_cols = 67, 68

# %
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

# # %%
# from models import tinyDarknet
# tinyDarknet_nn = tinyDarknet(x_train, y_train, num_class=1)
# tinyDarknet_nn.summary()

# %%
# from resnet import ResnetBuilder
#
# resnet_piccina = ResnetBuilder().build_resnet_18(input_shape=(1, 67, 68), num_outputs=1)
# resnet_piccina.summary()
# %%
# %
# #
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

model = energy_regressor_net

loss, std_err, std_error_log = train_adam_sgd(model,
                                              x_train, y_train, x_test, y_test,
                                              log_dir_tensorboard='test_dir',
                                              net_name='deepnet')

# %% Plot stuf
