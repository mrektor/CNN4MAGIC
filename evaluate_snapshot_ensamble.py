import pickle
from glob import glob

import numpy as np

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_2dense_energy

# %%

BATCH_SIZE = 512
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_energy=True,
                                                                     want_golden=False
                                                                     )
# %
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %%

net_name = 'MobileNetV2_2dense_energy_snap_whole'
files = glob(f'output_data/snapshots/{net_name}*2019-02-12*.h5')
files = sorted(files)
print(files[:-1])
# %%
print('Initializing NN...')
model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)

for idx, filepath in enumerate(files[:-1]):
    print('loading weights...')
    model.load_weights(filepath)

    print('Making predictions on test set...')
    y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
    print(f'saving {idx+2}...')
    with open(f'output_data/reconstructions/ensambels/pred_{net_name}_third_{filepath[-4]}', 'wb') as f:
        pickle.dump(y_pred, f)
    print('saved.')

print('All done. Everything OK')

