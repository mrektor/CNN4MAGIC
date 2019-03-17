import matplotlib

matplotlib.use('TkAgg')
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_separation

# %
BATCH_SIZE = 512
machine = '24cores'

# Load the datatr
train_gn, val_gn, test_gn = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=False,
    want_label=True,
    machine=machine,
    clean=True,
    include_time=False)

# Load the model
print('Loading the Neural Network...')
# model = MobileNetV2_separation(alpha=0.2, include_time=False)

# %%
import pickle

model = MobileNetV2_separation(alpha=1, include_time=False)
weights_path = '/data/new_magic/output_data/snapshots/MobileNetV2_separation_10_5_2019-03-11_22-00-11-Best.h5'
model.load_weights(weights_path)
y_pred_test = model.predict_generator(val_gn, workers=24, verbose=1, use_multiprocessing=True)

# %%
net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/point_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)

# %%
import pickle

net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/point_{net_name}.pkl'
with open(dump_name, 'rb') as f:
    point = pickle.load(f)
# %%
net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/crab_{net_name}.pkl'
with open(dump_name, 'rb') as f:
    crab = pickle.load(f)

# %%

import matplotlib.pyplot as plt

plt.figure()
plt.hist(point, bins=100, log=True, alpha=0.5)
plt.hist(crab, bins=100, log=True, alpha=0.5)
plt.legend(['Point-Like MC', 'Crab'])
plt.xlabel('Gammaness')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Point-like Montecarlo and Crab')
plt.savefig(f'output_data/pictures/Point_Crub_{net_name}_log.png')
plt.close()
