import pickle as pkl

from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import MobileNetV2_slim

# %%
# load IDs
filename = '/data2T/mariotti_data_2/npy_dump/train_val_test_dict_labels_list.pkl'
with open(filename, 'rb') as f:
    data, labels = pkl.load(f)

# %%
BATCH_SIZE = 64
train_gn = MAGIC_Generator(list_IDs=data['train'],
                           labels=labels,
                           batch_size=BATCH_SIZE,
                           )

val_gn = MAGIC_Generator(list_IDs=data['validation'],
                         labels=labels,
                         batch_size=BATCH_SIZE,
                         )

# %% Load Model
model = MobileNetV2_slim()
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %% Train
EPOCHS = 5

net_name = 'MobileNetV2_slim_fitgen'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=0.008,
                 num_epochs=5,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)
stop = EarlyStopping(patience=1)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=True,
                             workers=16
                             )
# %%
len(labels)
