import time

import pandas as pd

from CNN_Models.EnergyRegressor.models import *
from utils import *

# %
# Data Loading
x_train, y_train, x_test, y_test, input_shape = load_magic_data()

# Grid Search Definition
df = pd.DataFrame(columns=['network name', 'loss', 'std_error', 'wall time'])
# %%
# deep_magic_relu = deep_magic(input_shape, 'relu')

activations = ['softplus', 'relu', 'selu', 'elu', 'softsign']

log_dir_tensorboard = 'activations_search_run'
activation = 'softplus'
# for idx, activation in enumerate(activations):
model = deep_magic(input_shape=input_shape, activation=activation)
net_name = 'deep_magic_selu_classifier_cnn_' + activation

print(net_name)
before = time.time()
loss, std_err = train_adam_sgd(model, x_train, y_train, x_test, y_test, log_dir_tensorboard,
                               net_name, epochs=100, batch_size=200)
after = time.time()
wall_time = (after - before) / 60  # in minutes
idx = 0
df.loc[idx] = [net_name, loss, std_err, wall_time]
print('ACTUAL STATE OF DATAFRAME')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

# print('Saving the dataframe...')
# with open('pickle_data/dataframe_deep_magic_activation.pkl', 'wb') as f:
#     pickle.dump(df, f)
