import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import energy_skrr
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

# Load the model
print('Loading the Neural Network...')
model = energy_skrr(True)
net_name = 'energy_skrr_time'
# model = SEDenseNet121_energy_dropout_l2(drop=0)
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/energy_skrr_fromEpoch60_2019-03-13_12-36-30-Best.h5')
# net_name = 'SEDenseNet121_energy_dropout_l2'



result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.003,
                           epochs=50,
                           snapshot_number=15,
                           task='energy',
                           machine=machine,
                           swa=True
                           )
