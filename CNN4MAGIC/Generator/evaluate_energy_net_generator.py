# import matplotlib
#
# matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SE_InceptionV3_SingleDense_energy

# %
BATCH_SIZE = 128
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True, include_time=True,
    clean=False)

# %%
# Load the model
print('Loading the Neural Network...')
model = SE_InceptionV3_SingleDense_energy()
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/SE_InceptionV3_SingleDense_energy_yesTime_2019-03-16_19-49-37-Best.h5')
# %
net_name = 'SE_InceptionV3_SingleDense_energy_yestime_Best'

# %%
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# print(y_pred.shape)
# %

# net_name = 'single_DenseNet_piccina_Gold'
# filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/single_DenseNet_piccina_Gold.hdf5'
# model = load_model(filepath)
#
# %%
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=8)
# %%
import pickle

#%
file = f'output_data/reconstructions/energy_{net_name}.pkl'
with open(file, 'wb') as f:
    pickle.dump(y_pred, f)
#
#%
# with open(file, 'rb') as f:
#     y_pred = pickle.load(f)

# %%
import pickle

net_name = 'SE_InceptionV3_SingleDense_energy_yesTime_from60_SWA10'
file = f'/home/emariott/deepmagic/output_data/reconstructions/SE_InceptionV3_SingleDense_energy_yesTime_from60_2019-03-18_00-36-09.pkl'
with open(file, 'rb') as f:
    y_pred = pickle.load(f)

# %%
# net_name = ''
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name,
            fig_folder='output_data/pictures/energy_reconstruction',
            num_bins=100)
# %%
plot_gaussian_error(energy_te_limato, y_pred,
                    net_name=net_name,
                    fig_folder='output_data/pictures/energy_reconstruction', plot=False)
# %%

filename = '/home/emariott/deepmagic/output_data/reconstructions/energy_SEDenseNet121_energy_noclean_Gold_mse_from10_best_valloss0.2329.pkl'
with open(filename, 'rb') as f:
    y_pred = pickle.load(f)
#
#
import matplotlib.pyplot as plt

#
# plt.figure()
# sns.distplot(y_pred)
# plt.savefig('/home/emariott/ypred_net_1.png')
#
# # %%
# plt.figure()
# sns.distplot(y_pred)
# sns.distplot(energy_te_limato)
# plt.xlim([0,4])
# plt.savefig('/home/emariott/ee.png')
# plt.show()
#
# # %%
# import numpy as np
#
# a = np.array(e2)
# # %%
# a = np.array([v for k, v in e2.items()])
#
# # %%
plt.figure()
plt.hist(y_pred.flatten(), bins=1000)
# plt.hist(energy_te_limato, bins=100)
plt.xlim([0,4])
plt.savefig('/home/emariott/deepmagic/output_data/pictures/energy_diffuse.png')
#%%
import numpy as np

print(np.sum(y_pred > 5))
# %%
print(np.max(y_pred))

# %%
y_pred[y_pred > 5] = 5
# %%
import matplotlib.pyplot as plt

y_pred = y_pred[:len(energy_te_limato)]
plt.figure()
plt.hist2d(energy_te_limato, y_pred.flatten(), bins=400)
plt.savefig('output_data/pictures/hist2dtest.png')

#%%
plt.show()

#%%
