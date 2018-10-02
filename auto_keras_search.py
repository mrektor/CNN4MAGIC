from autokeras.image_supervised import ImageClassifier
import pickle

import numpy as np
from autokeras.image_supervised import ImageClassifier
from keras.models import load_model
from keras.utils import plot_model

with open('/dev/shm/data/hadron_numpy_train.pkl', 'rb') as f:
    hadron_tr = pickle.load(f)

with open('/dev/shm/data/gamma_numpy_train.pkl', 'rb') as f:
    gamma_tr = pickle.load(f)

x_train = np.concatenate((hadron_tr, gamma_tr))
y_train = np.concatenate((np.zeros(hadron_tr.shape[0]), np.ones(gamma_tr.shape[0])))

with open('/dev/shm/data/hadron_numpy_test.pkl', 'rb') as f:
    hadron_te = pickle.load(f)

with open('/dev/shm/data/gamma_numpy_test.pkl', 'rb') as f:
    gamma_te = pickle.load(f)

x_test = np.concatenate((hadron_te, gamma_te))
y_test = np.concatenate((np.zeros(hadron_te.shape[0]), np.ones(gamma_te.shape[0])))

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# %%
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

print(f'new shape for x_train {x_train.shape}')

clf = ImageClassifier(verbose=1, augment=False)
clf.fit(x_train, y_train, time_limit=60 * 60 * 1)
clf.load_searcher().load_best_model().produce_keras_model().save('my_model_pre.h5')

model = load_model('my_model_pre.h5')
plot_model(model, to_file='my_model_pre.png')

clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
clf.load_searcher().load_best_model().produce_keras_model().save('my_model_final.h5')

# %%
model = load_model('my_model_final.h5')  # See 'How to export keras models?' to generate this file before loading it.
plot_model(model, to_file='my_model_final.png')
