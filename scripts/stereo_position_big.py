import pickle

from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.snapshot import SnapshotCallbackBuilder
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

M = 5  # number of snapshots
nb_epoch = T = EPOCHS = 10  # number of epochs
alpha_zero = 0.0001  # initial learning rate
net_name = 'MobileNetV2_4dense_position-big-3'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
callbacks = snapshot.get_callbacks(model_prefix=net_name)

# %%
BATCH_SIZE = 128

print('Loading Data...')
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                    want_position=True,
                                                                    want_golden=False
                                                                    )

print('Loading the Neural Network...')
model = load_model('/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5')
model.compile(optimizer='sgd', loss='mse')
model.summary()

# %% Train

path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/' + net_name + '.hdf5'
# check = ModelCheckpoint(filepath=path, save_best_only=True)
# clr = OneCycleLR(max_lr=0.005,
#                  num_epochs=EPOCHS,
#                  num_samples=len(train_gn),
#                  batch_size=BATCH_SIZE)
# stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=callbacks,
                             use_multiprocessing=False,
                             workers=3)

prediction = model.predict_generator(generator=test_gn,
                                     use_multiprocessing=False,
                                     workers=3,
                                     verbose=1
                                     )

print('Saving prediction and result...')
dump_path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/position_predictions/pred_' + net_name + '.pkl'
with open(dump_path, 'wb') as f:
    pickle.dump(prediction, f)

dump_path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/position_predictions/result_' + net_name + '.pkl'
with open(dump_path, 'wb') as f:
    pickle.dump(result, f)

print('All done, everything went fine.')
