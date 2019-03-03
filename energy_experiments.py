import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import SEDenseNet121_energy
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 32
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    clean=False)

# Load the model
print('Loading the Neural Network...')
model = SEDenseNet121_energy()
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/SEDenseNet121_energy_noclean_Gold_2019-02-28_17-38-43-Best.h5')
net_name = 'SEDenseNet121_energy_noclean_Gold_mse'

result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.01,
                           epochs=10,
                           snapshot_number=8,
                           task='energy',
                           machine=machine,
                           swa=True
                           )
