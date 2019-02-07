from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import *

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %
print('Loading the Neural Network...')
model = MobileNetV2_2dense_energy(pretrained=False)
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
model.summary()

# % Train
EPOCHS = 30

net_name = 'MobileNetV2_2dense_energy GOLD'
path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True)
# %%
clr = OneCycleLR(max_lr=0.05,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)

stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=True,
                             workers=4)
# %%
