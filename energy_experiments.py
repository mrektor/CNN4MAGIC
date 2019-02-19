import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_4dense_energy
from CNN4MAGIC.Generator.evaluation_util import evaluate_energy
from CNN4MAGIC.Generator.training_util import superconvergence_training

BATCH_SIZE = 128

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_golden=True,
                                                                  want_energy=True,
                                                                  folder_diffuse='/ssdraptor/magic_data/data_processed/diffuse',
                                                                  folder_point='/ssdraptor/magic_data/data_processed/point_like')

# Load the model
print('Loading the Neural Network...')
model = MobileNetV2_4dense_energy(pretrained=False, drop=False, freeze_cnn=False)
# model.load_weights(
#     '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_11_2019-02-17_01-38-48-5.h5')
net_name = 'MobileNetV2_4dense_energy_superconvergence'

# Train
result, y_pred = superconvergence_training(model=model, net_name=net_name,
                                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                                           batch_size=BATCH_SIZE,
                                           patience=5,
                                           epochs=30,
                                           max_lr=0.45)

# result, y_pred = snapshot_training(model=model,
#                                    train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
#                                    net_name=net_name,
#                                    max_lr=0.05,
#                                    epochs=5,
#                                    snapshot_number=5
#                                    )

# Evaluate
evaluate_energy(energy, y_pred, net_name)
