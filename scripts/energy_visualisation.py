import pickle

from CNN4MAGIC.Generator.evaluation_util import evaluate_energy
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_2dense_energy

# %
# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_golden=False,
                                                                  want_energy=True)
# %%
model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)
model.load_weights(
    '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_2019-02-08_14-59-54-5.h5')
net_name = 'MobileNetV2_2dense_energy_snap_whole_epoch5'

y_pred_test = model.predict_generator(test_gn, workers=3, verbose=1)

reconstructions_path = f'output_data/reconstructions/{net_name}.pkl'
with open(reconstructions_path, 'wb') as f:
    pickle.dump(y_pred_test, f)

# Evaluate
evaluate_energy(energy, y_pred_test, net_name)

# %%
with open('/home/emariott/deepmagic/output_data/reconstructions/MobileNetV2_2dense_energy_snap_whole_epoch5.pkl',
          'rb') as f:
    y_pred = pickle.load(f)

# %

net_name = 'MobileNetV2_2dense_energy_snap_whole_epoch5'
evaluate_energy(energy, y_pred, net_name)
