import numpy as np

from CNN4MAGIC.CNN_Models.BigData.snapshot import SnapshotCallbackBuilder
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import single_DenseNet_25_3_doubleDense

M = 5  # number of snapshots
nb_epoch = T = 10  # number of epochs
alpha_zero = 0.03  # initial learning rate
net_name = 'single_DenseNet_25_3_doubleDense_snap'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
callbacks = snapshot.get_callbacks(model_prefix=net_name)

print('Loading the Neural Network...')
model = single_DenseNet_25_3_doubleDense()
model.compile(optimizer='sgd', loss='mse')
model.summary()
print('Model Loaded.')

# %%
BATCH_SIZE = 512
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                    want_energy=True,
                                                                    want_golden=False
                                                                    )
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

print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)

energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name, fig_folder='/data/code/CNN4MAGIC/Generator/energy_pic/',
            num_bins=100)
# %%
plot_gaussian_error(energy_te_limato, y_pred, net_name=net_name,
                    fig_folder='/data/code/CNN4MAGIC/Generator/energy_pic/')
print('All done.')
