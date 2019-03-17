import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SE_InceptionV3_SingleDense_energy
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 128
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    include_time=True,
    clean=False)
# %%
# Load the model
print('Loading the Neural Network...')
model = SE_InceptionV3_SingleDense_energy(True)
net_name = 'SE_InceptionV3_SingleDense_energy_yesTime'
# model = SEDenseNet121_energy_dropout_l2(drop=0)
# %%
# model.load_weights(
#     'output_data/snapshots/SE_InceptionV3_DoubleDense_energy_2019-03-15_01-15-55-6.h5')
# net_name = 'SEDenseNet121_energy_dropout_l2'

#%%
result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.05,
                           epochs=40,
                           snapshot_number=15,
                           task='energy',
                           machine=machine,
                           swa=True
                           )

# res = model.evaluate_generator(val_gn, verbose=1, use_multiprocessing=True, workers=8)
