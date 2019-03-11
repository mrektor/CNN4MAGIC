import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SEDenseNet121_energy_dropout_l2
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 32
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    clean=False)

# Load the model
print('Loading the Neural Network...')
model = SEDenseNet121_energy_dropout_l2(drop=0)
# model.load_weights(
#     'output_data/snapshots/SEDenseNet121_position_noclean_Gold_fromEpoch35_2019-03-04_17-03-30-Best.h5')
net_name = 'SEDenseNet121_energy_dropout_l2'

result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.003,
                           epochs=20,
                           snapshot_number=15,
                           task='energy',
                           machine=machine,
                           swa=True
                           )
