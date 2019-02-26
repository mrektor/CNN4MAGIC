import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_separation
from CNN4MAGIC.Generator.training_util import snapshot_training

BATCH_SIZE = 1024
machine = '24cores'

# Load the data
train_gn, val_gn = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=False,
    want_label=True,
    machine=machine,
    clean=True)

# Load the model
print('Loading the Neural Network...')
model = MobileNetV2_separation()
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_11_2019-02-17_01-38-48-5.h5')
net_name = 'MobileNetV2_separation_clean6_3punto5_Gold'

result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn,
                           net_name=net_name,
                           machine=machine,
                           max_lr=0.05,
                           epochs=15,
                           snapshot_number=10,
                           do_telegram=True
                           )
