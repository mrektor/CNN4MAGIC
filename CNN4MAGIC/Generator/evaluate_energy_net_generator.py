from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE, want_energy=True)

# %
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %
print(len(test_gn) * BATCH_SIZE)
print(energy_te.shape)
print(energy_te_limato.shape)
# %

# net_name = 'MobileNetV2_4dense_energy_dropout_GOLD'
# filepath = '/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_energy_dropout_GOLD'
# model = load_model(filepath)
#
# # %
# print('Making predictions on test set...')
# y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=3)
# %%
import pickle

file = '/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/pred_MobileNetV2_4dense_energy_pretreained_freeze.pkl'
with open(file, 'rb') as f:
    y_pred = pickle.load(f)


# %%
net_name = 'MobileNetV2_4dense_energy_pretreained_freeze'
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error

plot_hist2D(energy_te_limato, y_pred, net_name,
            fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/',
            num_bins=100)
# %%
plot_gaussian_error(energy_te_limato, y_pred, net_name=net_name
                    ,
                    fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/energy_gen_stuff/')
# %%
