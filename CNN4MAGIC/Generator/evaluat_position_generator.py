# %%
from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_position

BATCH_SIZE = 128
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

model = MobileNetV2_4dense_position()
print('Loading weights...')
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_4dense_position_beast_2019-02-19_18-40-38-Best.h5')

print('start predictions...')
position_prediction = model.predict_generator(test_gn, verbose=1)

train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True,
    folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
    folder_point='/ssdraptor/magic_data/data_processed/point_like')

position_te_limato = position[:position_prediction.shape[0], :]
energy_te_limato = energy[:position_prediction.shape[0]]
# %

print(position.shape, position_prediction.shape, position_te_limato.shape)
print(energy.shape, energy_te_limato.shape)
# %%
net_name = 'MobileNetV2-4D 30 epochs best'
plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name,
                        fig_folder='/home/emariott/deepmagic/output_data/pictures/direction_reconstruction')
