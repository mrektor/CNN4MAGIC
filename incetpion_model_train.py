import time

import pandas as pd
from keras.losses import *

from magic_inception import magic_inception
from utils import *

train = True
input_shape = (67, 68, 1)
if train:
    x_train, y_train, x_test, y_test, input_shape = load_magic_data()
    print(input_shape)

# %%
log_dir_tensorboard = 'manual_filt_gridsearch'
input_shape = (67, 68, 1)
# num_filt = 2 * 3 * 15 # 13 bello, 15 fighissimo
num_filts = [2 * 3 * i for i in range(4, 25)]
df = pd.DataFrame(columns=['wall time', 'num_filt', 'loss', 'std'])
for num_filt in num_filts:
    model = magic_inception(input_shape, num_filt, dropout=0, do_res=False)
    # model = deep_magic(input_shape, 'relu')
    net_name = 'magic_inception_varying_filts_' + str(num_filt)
    model.summary()
    # %%

    print(net_name)
    before = time.time()
    loss, std_err = train_adam(model, x_train, y_train, x_test, y_test, log_dir_tensorboard,
                               net_name, custom_loss=mean_absolute_error, epochs=100, batch_size=128,
                               initial_lr=0.0005)
    after = time.time()
    wall_time = (after - before) / 60  # in minutes
    df = df.append({'wall time': wall_time, 'std': std_err, 'num_filt': num_filt, 'loss': loss}, ignore_index=True)
    print(df)

with open('pickle_data/df_manual_filt_gridsearch.pkl', 'wb') as f:
    pickle.dump(df, f, protocol=4)
