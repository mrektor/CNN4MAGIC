from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import *

BATCH_SIZE = 400
train_gn, val_gn, position = load_data_generators(batch_size=BATCH_SIZE, want_position=True)

# %%
print('Loading the Neural Network...')
model = MobileNetV2_position()
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
model.summary()

# %% Train
EPOCHS = 30

net_name = 'MobileNetV2-alpha1-position-buonanno'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=1e-4,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)
stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=False,
                             workers=1,
                             max_queue_size=30
                             )
