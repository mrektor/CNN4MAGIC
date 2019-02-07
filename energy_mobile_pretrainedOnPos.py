import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import *

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy, e2 = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %
print('Loading the Neural Network...')
model = single_DenseNet_piccina()
model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
model.summary()

# % Train
EPOCHS = 30

net_name = 'single_DenseNet_piccina_Gold'
path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/' + net_name + '.hdf5'
check = ModelCheckpoint(filepath=path, save_best_only=True, period=1)
# %e
clr = OneCycleLR(max_lr=0.5,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)

stop = EarlyStopping(patience=5)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=False,
                             workers=8)

print('Finished training, start prediction')
# %%
prediction = model.predict_generator(generator=test_gn,
                                     use_multiprocessing=False,
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
