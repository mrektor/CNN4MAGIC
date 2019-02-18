import pickle

import numpy as np

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point
from CNN4MAGIC.Generator.models import MobileNetV2_2dense_energy

# from CNN4MAGIC.Generator.evaluation_util import evaluate_energy

# %

BATCH_SIZE = 128
train_gn, val_gn, test_gn, energy_te = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_energy=True,
                                                                     want_golden=True
                                                                     )
# %
energy_te = np.array(energy_te)
energy_te_limato = energy_te[:len(test_gn) * BATCH_SIZE]

# %

# %
with open(
        '/home/emariott/deepmagic/output_data/reconstructions/ensambels/pred_MobileNetV2_2dense_energy_snap_whole_11_fourth_5',
        'rb') as f:
    pred = pickle.load(f)

# %
from CNN4MAGIC.Generator.evaluation_util import evaluate_energy

net_name = 'MobileNetV2_2dense_energy_snap_whole_11_fourth_gaus'

evaluate_energy(energy_te, pred, net_name)

# %%
net_name = 'MobileNetV2_2dense_energy_snap_whole_11'
model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)
filepath = '/home/emariott/deepmagic/output_data/snapshots/MobileNetV2_2dense_energy_snap_whole_11_2019-02-17_01-38-48-5.h5'
print('loading weights...')
model.load_weights(filepath)

print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
print(f'saving...')
with open(f'output_data/reconstructions/ensambels/pred_{net_name}_fourth_{filepath[-4]}', 'wb') as f:
    pickle.dump(y_pred, f)
print('saved.')
# evaluate_energy(energy_te, y_pred, net_name)


# %%


# net_name = 'MobileNetV2_2dense_energy_snap_whole'
# files = glob(f'output_data/snapshots/{net_name}*2019-02-12*.h5')
#
# files = sorted(files)
# print(files[:-1])
# # %%
# print('Initializing NN...')
# model = MobileNetV2_2dense_energy(pretrained=True, drop=False, freeze_cnn=False)
#
# for idx, filepath in enumerate(files[:-1]):
#     print('loading weights...')
#     model.load_weights(filepath)
#
#     print('Making predictions on test set...')
#     y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=False, workers=3)
#     print(f'saving {idx+2}...')
#     with open(f'output_data/reconstructions/ensambels/pred_{net_name}_third_{filepath[-4]}', 'wb') as f:
#         pickle.dump(y_pred, f)
#     print('saved.')
#
# print('All done. Everything OK')
