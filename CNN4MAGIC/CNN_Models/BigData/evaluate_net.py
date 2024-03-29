import numpy as np
from keras.models import load_model

from CNN4MAGIC.CNN_Models.BigData.loader import load_data_append
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

net_name = 'single_DenseNet_25_3_doubleDense-noImpact'

path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints/' + net_name + '.hdf5'
print('Loading model ' + net_name + '...')
model = load_model(path)
m1_te, m2_te, energy_te = load_data_append('test', prune=True, leak=True)
y_test = np.log10(energy_te)

print('Making Predictions...')
y_pred = model.predict({'m1': m1_te, 'm2': m2_te})

# %%
net_name = 'single_DenseNet_25_3_doubleDense_noImpactCutTrain-YesImpactCutTest'

print('Plotting stuff...')
plot_hist2D(y_test, y_pred,
            fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/',
            net_name=net_name,
            num_bins=100)

# %%
plot_gaussian_error(y_test, y_pred,
                    net_name=net_name + '_13bin',
                    num_bins=13,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/')
print('All done')