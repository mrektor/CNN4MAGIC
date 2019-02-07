import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import *

BATCH_SIZE = 64
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %
print('Loading the Neural Network...')
model = MobileNetV2_4dense_energy_dropout()
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
model.summary()

# % Train
EPOCHS = 30

net_name = 'MobileNetV2_4dense_energy_dropout_GOLD'
path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True, period=1)
# %%
clr = OneCycleLR(max_lr=0.04,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)

stop = EarlyStopping(patience=3)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=True,
                             workers=3)

print('Finished training, start prediction')
# %%
prediction = model.predict_generator(generator=test_gn,
                                     use_multiprocessing=True,
                                     workers=3,
                                     verbose=1
                                     )

print('Saving prediction and result...')
dump_path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/pred_' + net_name + '.pkl'
with open(dump_path, 'wb') as f:
    pickle.dump(prediction, f)

dump_path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/result_' + net_name + '.pkl'
with open(dump_path, 'wb') as f:
    pickle.dump(result, f)

print('All done, everything went fine.')
