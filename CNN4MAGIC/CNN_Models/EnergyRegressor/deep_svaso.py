import time

import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from utils import *


# %%
def se_block(x, f, rate=2):
    m = GlobalAveragePooling2D()(x)
    m = Dense(f // rate)(m)
    m = Dense(f, activation='sigmoid')(m)
    return multiply([x, m])


def common_blocks(x, num_filter):
    x = Conv2D(num_filter, (3, 3), padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(num_filter, (1, 1), padding='valid')(x)
    x = ReLU()(x)
    # x = se_block(x, 32)
    # x = BatchNormalization(center=False, scale=False)(x)
    return x


def svaso(x, num_neurons, num_filt=10):
    l = []
    m = np.random.poisson(2, num_neurons) + 1
    n = np.random.poisson(2, num_neurons) + 1
    for i in range(num_neurons):
        k = (m[i], n[i])
        d = (1, 1)
        o = Conv2D(num_filt, kernel_size=k, dilation_rate=d, padding='same')(x)
        o = common_blocks(o)
        # o = se_block(o, num_filt)
        l.append(o)
    # [l(i) for i in range()]
    final = concatenate(l)
    # print(l)
    return final


def dense_block(x, num_filt=5, d=5):
    l = x
    for i in range(d):
        x = common_blocks(x, num_filt)
        l = concatenate([l, x])
    return l


# %%
def svasonet(input_shape, numclasses):
    input = Input(input_shape)
    x = Conv2D(64, (5, 5), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(2):
        x = common_blocks(x, 86)
    x = MaxPooling2D()(x)

    for _ in range(2):
        x = common_blocks(x, 128)
    x = MaxPooling2D()(x)

    for _ in range(1):
        x = common_blocks(x, 150)
    # x = MaxPooling2D()(x)

    # dense = dense_block(link, num_filt=64, d=5)
    # link2 = MaxPooling2D(pool_size=(3, 3))(dense)
    # dense2 = dense_block(link2, num_filt=50, d=5)
    # link3 = MaxPooling2D(pool_size=(3, 3))(dense2)
    # dense3 = dense_block(link3, num_filt=50, d=2)

    link4 = GlobalAveragePooling2D()(x)
    # flat = Flatten()(link2)
    out = Dense(50, activation='relu')(link4)
    out = Dense(numclasses, activation='sigmoid')(out)
    svaso_net = Model(inputs=input, outputs=out)
    return svaso_net


# %%

###### TRAIN IT ###########
print('Loading data...')
bef = time.time()
x_train, y_train, x_test, y_test, input_shape = load_magic_data()
num_classes = 15

values_bin_train, bins = bin_data(y_train, num_bins=num_classes)
values_bin_test, _ = bin_data(y_test, num_bins=num_classes, bins=bins)

encoder = LabelEncoder()
y_train_cat = encoder.fit_transform(values_bin_train)
y_test_cat = encoder.transform(values_bin_test)

y_train_keras_cat = to_categorical(y_train_cat, num_classes=num_classes)
y_test_keras_cat = to_categorical(y_test_cat, num_classes=num_classes)
print(f'Data loaded and processed in {time.time()-bef} seconds')
# %%
log_dir_tensorboard = 'dense'
df = pd.DataFrame(columns=['wall time', 'loss', 'std'])

for i in range(1):
    print('..Building the network svasonet....')
    bef = time.time()
    model = svasonet(input_shape, numclasses=num_classes)
    print()
    print(f'it took {(time.time()-bef)/60} minutes to build the net')
    print()
    model.summary()
    net_name = 'svasonet_dense_' + str(num_classes)
    # %
    batch_size = 256
    epochs = 30

    # before = time.time()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    early_stop = EarlyStopping(patience=5, min_delta=0.0001)
    nan_stop = TerminateOnNaN()
    check = ModelCheckpoint('/data/mariotti_data/checkpoints/grid_inc_filt_' + net_name + '.hdf5', period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=4, min_lr=0.000005)

    result = model.fit(x_train, y_train_keras_cat,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(x_test, y_test_keras_cat),
                       callbacks=[early_stop, nan_stop, reduce_lr, check])

    after = time.time()
    # wall_time = (after - before) / 60  # in minutes
    # df = df.append({'wall time': wall_time, 'std': std_err, 'loss': loss}, ignore_index=True)
    # print(df)

    ### PRINT FIGURES
    y_pred_hot = model.predict(x_test)

    num_bins = bins.shape[0]
    bins_mean_value = np.zeros(num_bins - 1)
    for i in range(num_bins - 1):
        bins_mean_value[i] = np.median([bins[i], bins[i + 1]])

    # %
    cm = confusion_matrix(y_true=np.argmax(y_test_keras_cat, 1), y_pred=np.argmax(y_pred_hot, 1))
    plt.figure()
    plot_confusion_matrix(cm=cm, classes=np.around(bins_mean_value, 2), normalize=True)
    plt.savefig('/data/mariotti_data/pics/confusion_matrix_' + net_name + '.jpg')

# with open('pickle_data/df_manual_filt_gridsearch.pkl', 'wb') as f:
#     pickle.dump(df, f, protocol=4)
