from CNN4MAGIC.CNN_Models.BigData.snapshot import SnapshotCallbackBuilder
from CNN4MAGIC.Generator.gen_util import load_data_generators
from CNN4MAGIC.Generator.models import MobileNetV2_2dense_energy

M = 5  # number of snapshots
nb_epoch = T = 10  # number of epochs
alpha_zero = 0.03  # initial learning rate
net_name = 'MobileNetV2_2dense_energy_snap'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
callbacks = snapshot.get_callbacks(model_prefix=net_name)

print('Loading the Neural Network...')
model = MobileNetV2_2dense_energy()
model.compile(optimizer='sgd', loss='mse')
model.summary()
print('Model Loaded.')

# %%
BATCH_SIZE = 512
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

# %% Train
result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=nb_epoch,
                             verbose=1,
                             callbacks=callbacks,
                             use_multiprocessing=True,
                             workers=24
                             )

# %%
print('Training done.')
