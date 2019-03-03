# import matplotlib
#
# matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SEDenseNet121_energy

BATCH_SIZE = 32
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    clean=False)

# %%
# Load the model
print('Loading the Neural Network...')
model = SEDenseNet121_energy()
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/SEDenseNet121_energy_noclean_Gold_2019-02-28_17-38-43-Best.h5')
# %%
net_name = 'SEDenseNet121_energy_noclean_Gold_FastSWA_maybe'

# %
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# %%

# net_name = 'single_DenseNet_piccina_Gold'
# filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/single_DenseNet_piccina_Gold.hdf5'
# model = load_model(filepath)
#
# # %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
# %%
import pickle

file = f'output_data/reconstructions/energy_{net_name}.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(y_pred, f)
with open(file, 'rb') as f:
    y_pred = pickle.load(f)

# %%
# net_name = 'MobileNetV2_2dense_energy_pretrained'
# file = f'/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/pred_{net_name}.pkl'
# with open(file, 'rb') as f:
#     y_pred = pickle.load(f)


# %%
# net_name = ''
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name,
            fig_folder='/home/emariott/deepmagic/output_data/pictures/energy_reconstruction/',
            num_bins=100)
# %%
plot_gaussian_error(energy_te_limato, y_pred,
                    net_name=net_name + 'ye',
                    fig_folder='/home/emariott/deepmagic/output_data/pictures/energy_reconstruction/')
# %%


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.distplot(y_pred)
plt.savefig('/home/emariott/ypred_net_1.png')

# %%
plt.figure()
sns.distplot(y_pred)
sns.distplot(energy_te_limato)
plt.xlim([0,4])
plt.savefig('/home/emariott/ee.png')
plt.show()

# %%
import numpy as np

a = np.array(e2)
# %%
a = np.array([v for k, v in e2.items()])

# %%
sns.distplot(y_pred)
sns.distplot(energy_te_limato)
sns.distplot(a)
plt.savefig('/home/emariott/energy_diffuse.png')
plt.show()
