# %%

from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import NASNet_mobile_position

BATCH_SIZE = 1024
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

model = NASNet_mobile_position()
print('Loading weights...')
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/NASNet_mobile_position_2019-02-20_18-04-55-Best.h5')


# model = load_model('/home/emariott/deepmagic/output_data/checkpoints/MV2-4D-30E-l2-EnsLast9_2019-02-20_11-28-13.hdf5')
# print('start predictions...')

position_prediction = model.predict_generator(test_gn, verbose=1)

train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

# %
net_name = 'NasNet-40Epoch-best'

# with open(f'/home/emariott/deepmagic/output_data/reconstructions/pred_{net_name}_position.pkl', 'rb') as f:
#     position_prediction = pickle.load(f)

position_te_limato = position[:position_prediction.shape[0], :]
energy_te_limato = energy[:position_prediction.shape[0]]
# %

print(position.shape, position_prediction.shape, position_te_limato.shape)
print(energy.shape, energy_te_limato.shape)

# %%
# import pickle
#
# with open(f'/home/emariott/deepmagic/output_data/reconstructions/pred_{net_name}_position.pkl', 'wb') as f:
#     pickle.dump(position_prediction, f)
# %%


net_name = 'NasNet-40Epoch-best'
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name,
                        fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction')
