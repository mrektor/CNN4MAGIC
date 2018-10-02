# %%
from __future__ import print_function

import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

# %%

with open('/data/data/hadron_numpy_train.pkl', 'rb') as f:
    hadron_tr = pickle.load(f)

with open('/data/data/gamma_numpy_train.pkl', 'rb') as f:
    gamma_tr = pickle.load(f)

x_train = np.concatenate((hadron_tr, gamma_tr))
y_train = np.concatenate((np.zeros(hadron_tr.shape[0]), np.ones(gamma_tr.shape[0])))

with open('/data/data/hadron_numpy_test.pkl', 'rb') as f:
    hadron_te = pickle.load(f)

with open('/data/data/gamma_numpy_test.pkl', 'rb') as f:
    gamma_te = pickle.load(f)

x_test = np.concatenate((hadron_te, gamma_te))
y_test = np.concatenate((np.zeros(hadron_te.shape[0]), np.ones(gamma_te.shape[0])))

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 101, 101

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
sample = hadron_te[0, :, :]
print(sample.shape)
plt.imshow(sample)
plt.show()
# %
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# #%%
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
#
# #%%
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
