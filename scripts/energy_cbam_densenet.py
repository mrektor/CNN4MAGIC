from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_energy_dropout

print('Loading the Neural Network...')
model = MobileNetV2_4dense_energy_dropout()
model.compile(optimizer='sgd', loss='mse')
model.summary()
print('Model Loaded.')


#%%
BATCH_SIZE = 256
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)


# %% Train
EPOCHS = 30

net_name = 'MobileNetV2_4dense_energy_drop03'
path = '/data/code/CNN4MAGIC/Generator/checkpoints/' + net_name + '.hdf5'
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=0.08,
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
                             workers=24
                             )

# %%
print('Training done.')
