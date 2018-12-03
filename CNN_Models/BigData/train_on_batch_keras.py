import glob
import pickle
import random

import keras
import matplotlib.pyplot as plt
import numpy as np

# %%
fileList = glob.glob('/data2T/mariotti_data_2/interp_from_root/MC/*.pkl')
random.shuffle(fileList)

# %%
trainList = fileList[:int(len(fileList) * 0.5)]
valList = fileList[int(len(fileList) * 0.5):int(len(fileList) * 0.75)]
testList = fileList[int(len(fileList) * 0.75):]

# %% model definition
from keras.layers import Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU


def feats(x_train, input_shape, baseDim=16, padding="valid", dropout=0.1):
    out = Conv2D(baseDim * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding,
                 input_shape=input_shape)(x_train)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    # out = MaxPooling2D(pool_size=(2, 2)(out))

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    # out = MaxPooling2D(pool_size=(2, 2)(out))

    out = Dropout(dropout)(out)
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)

    out = Conv2D(baseDim * 8 * 2 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 8 * 2 * 2, (3, 3), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)
    out = ReLU()(out)
    #
    out = Conv2D(baseDim * 2 * 2, (1, 1), strides=(1, 1), use_bias=False, padding=padding)(out)
    out = BatchNormalization(epsilon=1e-05, momentum=0.1)(out)

    out = GlobalAveragePooling2D()(out)

    return out


from keras.layers import Conv2D, MaxPooling2D, Input, Dense
from keras.models import Model

m1 = Input(shape=(67, 68, 2))
m2 = Input(shape=(67, 68, 2))
# %%
out_a = feats(m1, (67, 68, 2))
out_b = feats(m2, (67, 68, 2))

concatenated = keras.layers.concatenate([out_a, out_b])

out = Dense(60, activation='relu')(concatenated)
out = Dropout(0.2)(out)
out = Dense(20, activation='relu')(out)
out = Dense(1)(out)

energy_regressor = Model(inputs=[m1, m2], outputs=out)

energy_regressor.compile(optimizer='adam', loss='mse')
energy_regressor.summary()

# %%
num_epochs = 5
tr_losses = []
te_losses = []
for epoch in range(num_epochs):

    random.shuffle(trainList)
    print('=============')
    print(f'Epoch {epoch}')
    print('=============')

    for batch in trainList:
        with open(batch, 'rb') as f:
            data = pickle.load(f)
        m1interp = data['M1_interp'].reshape(data['M1_interp'].shape[0], 67, 68, 2)  # Channel last porca puttana
        m2interp = data['M2_interp'].reshape(data['M2_interp'].shape[0], 67, 68, 2)  # Channel last porca puttana

        loss = energy_regressor.train_on_batch(x=[m1interp, m2interp], y=data['energy'])
        tr_losses.append(loss)
        print(f'Log Train loss: {np.log10(loss)}')

    for batch in valList:
        with open(batch, 'rb') as f:
            data = pickle.load(f)
        m1interp = data['M1_interp'].reshape(data['M1_interp'].shape[0], 67, 68, 2)  # Channel last porca puttana
        m2interp = data['M2_interp'].reshape(data['M2_interp'].shape[0], 67, 68, 2)  # Channel last porca puttana
        loss = energy_regressor.test_on_batch(x=[m1interp, m2interp], y=data['energy'])
        te_losses.append(loss)
        print(f'Log Loss on validation set: {np.log10(loss)}')

    energy_regressor.save('/data2T/mariotti_data_2/stereo_models/energy_regressor.h5')
    plt.plot(tr_losses)
    plt.plot()
