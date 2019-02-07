from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te, e2 = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# %

net_name = 'single_DenseNet_piccina_Gold'
filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/single_DenseNet_piccina_Gold.hdf5'
model = load_model(filepath)

# %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
# %%
# import pickle
#
# file = '/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/pred_MobileNetV2_4dense_energy_pretreained_freeze.pkl'
# with open(file, 'rb') as f:
#     y_pred = pickle.load(f)


# %%
# net_name = ''
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name,
            fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/',
            num_bins=100)
# %%
plot_gaussian_error(energy_te_limato, y_pred, net_name=net_name
                    ,
                    fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/')
# %%


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(y_pred)
plt.savefig('/home/emariott/ypred_net.png')

# %%
sns.distplot(y_pred)
sns.distplot(energy_te_limato)
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
