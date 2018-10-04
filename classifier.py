# %%
from __future__ import print_function

import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import roc_curve, auc, confusion_matrix

from utils import plot_confusion_matrix

# %% Data Loading

with open('pickle_data/hadron_numpy_train.pkl', 'rb') as f:
    hadron_tr = pickle.load(f)

with open('pickle_data/gamma_numpy_train.pkl', 'rb') as f:
    gamma_tr = pickle.load(f)

x_train = np.concatenate((hadron_tr, gamma_tr))
y_train = np.concatenate((np.zeros(hadron_tr.shape[0]), np.ones(gamma_tr.shape[0])))

with open('pickle_data/hadron_numpy_test.pkl', 'rb') as f:
    hadron_te = pickle.load(f)

with open('pickle_data/gamma_numpy_test.pkl', 'rb') as f:
    gamma_te = pickle.load(f)

x_test = np.concatenate((hadron_te, gamma_te))
y_test = np.concatenate((np.zeros(hadron_te.shape[0]), np.ones(gamma_te.shape[0])))

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

batch_size = 128
num_classes = 2
epochs = 5

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %% Check Data
# idx = np.random.randint(0, 10000)
# sample = hadron_tr[idx, :, :]
# print(sample.shape)
# plt.figure()
# plt.imshow(sample)
# plt.colorbar()
# plt.savefig('random_hadron.jpg')

# %%
# %
#
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# %% Handeld Definition
tensorboard = TensorBoard()
early_stop = EarlyStopping(patience=2)
check = ModelCheckpoint('checkpoints/check.hdf5')

epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard, early_stop, check])

# %%

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% ROC CURVE
y_pred_keras = model.predict_proba(x_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:, 1], y_pred_keras[:, 1])
auc_keras = auc(fpr_keras, tpr_keras)

# %% Plot ROC
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='CNN (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('pics/ROC.jpg')
plt.show()

plt.figure()
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='CNN (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.savefig('pics/ROCzoom.jpg')
plt.show()

# %% Plot Confusion Matrix
y_pred = model.predict(x_test)
cnf_matrix = confusion_matrix(y_test[:, 1], np.round(y_pred[:, 1]))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['Hadrons', 'Gammas']
plot_confusion_matrix(cnf_matrix,
                      classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('pics/ConfusionMatrix.jpg')

# %% Plot Errors
mistaken = []
print('The index of misclassified items are:')
for idx in range(y_test.shape[0]):
    if y_test[idx] != np.round(y_pred[idx]):
        print(idx)
        mistaken.append(idx)
# %%
for idx in mistaken:
    plt.figure()
    plt.imshow(x_test[idx, :, :, 0])
    plt.colorbar()
    plt.title('Mistakenly Classified')
    plt.savefig('pics/err' + str(idx) + '.jpg')
