import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.training_util import snapshot_training
from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2

# %%
BATCH_SIZE = 32
machine = 'titanx'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    include_time=True,
    clean=False)
#%%
# Load the model
print('Loading the Neural Network...')
model = SEDenseNet121_position_l2(include_time=True)
# model = load_model(
#     '/home/emariott/software_magic/output_data/checkpoints/SE-121-Position-TransferEnsemble5-from59to63.hdf5')
model.load_weights(
    '/home/emariott/software_magic/output_data/snapshots/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27-Best.h5')
net_name = 'SE-121-Position-l2-fromepoch80'
#%%
# Train
# result, y_pred = superconvergence_training(model=model, net_name=net_name,
#                                            train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
#                                            batch_size=BATCH_SIZE,
#                                            patience=5,
#                                            epochs=30,
#                                            max_lr=0.45)

result = snapshot_training(model=model,
                           machine=machine,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.00075,
                           epochs=7,
                           snapshot_number=7,
                           task='direction',
                           swa=1
                           )

# Evaluate

# train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
#     batch_size=BATCH_SIZE,
#     want_golden=True,
#     want_energy=True,
#     folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
#     folder_point='/ssdraptor/magic_data/data_processed/point_like')
#
# position_te_limato = position[:position_prediction.shape[0], :]
# energy_te_limato = energy[:position_prediction.shape[0]]
# # %%
# plot_angular_resolution(position_te_limato, position_prediction, energy_te_limato, net_name=net_name)
