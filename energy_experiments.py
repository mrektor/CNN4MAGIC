from CNN4MAGIC.Generator.evaluation_util import evaluate_energy
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import *
from CNN4MAGIC.Generator.training_util import superconvergence_training

BATCH_SIZE = 128

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_golden=True,
                                                                  want_energy=True)

# Load the model
print('Loading the Neural Network...')
model = MobileNetV2_4dense_energy(pretrained=True, drop=False, freeze_cnn=False)
net_name = 'MobileNetV2_4dense_energy_Gold'

# Train
result, y_pred = superconvergence_training(model=model, net_name=net_name,
                                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                                           batch_size=BATCH_SIZE,
                                           epochs=3,
                                           max_lr=0.3)

# Evaluate
evaluate_energy(energy, y_pred, net_name)
