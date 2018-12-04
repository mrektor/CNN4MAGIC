import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Dropout, GlobalAveragePooling2D, BatchNormalization, ReLU

from CNN4MAGIC.CNN_Models.BigData.loader import load_data_train, load_data_val

m1_tr, m2_tr, energy_tr = load_data_train()
m1_val, m2_val, energy_val = load_data_val()

energy_tr = np.log10(energy_tr)
energy_val = np.log10(energy_val)


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

m1 = Input(shape=(67, 68, 2), name='m1')
m2 = Input(shape=(67, 68, 2), name='m2')
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

net_name = 'EnergyRegressorStereoTime'
early_stop = EarlyStopping(patience=8, min_delta=0.0001)
nan_stop = TerminateOnNaN()
check = ModelCheckpoint('/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints' + net_name + '.hdf5', period=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                              patience=4, min_lr=0.000005)

result = energy_regressor.fit({'m1': m1_tr, 'm2': m2_tr}, energy_tr,
                              batch_size=128,
                              epochs=100,
                              verbose=1,
                              validation_data=({'m1': m1_val, 'm2': m2_val}, energy_val),
                              callbacks=[early_stop, nan_stop, reduce_lr, check])
