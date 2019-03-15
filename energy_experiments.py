import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SE_InceptionV3_DoubleDense_energy
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 64
machine = 'titanx'

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
model = SE_InceptionV3_DoubleDense_energy()
net_name = 'SE_InceptionV3_DoubleDense_energy'
# model = SEDenseNet121_energy_dropout_l2(drop=0)
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/energy_skrr_fromEpoch60_2019-03-13_12-36-30-Best.h5')
# net_name = 'SEDenseNet121_energy_dropout_l2'


result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.1,
                           epochs=13,
                           snapshot_number=5,
                           task='energy',
                           machine=machine,
                           swa=True
                           )
