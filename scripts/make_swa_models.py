from glob import glob

import numpy as np
from tqdm import tqdm

from CNN4MAGIC.Generator.models import SEDenseNet121_position_l2


# %%

def compute_SWA(weights, model):
    # weights = [model.get_weights() for model in model_list]

    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            [np.array(weights_).mean(axis=0) \
             for weights_ in zip(*weights_list_tuple)])

    new_model = model
    new_model.set_weights(new_weights)
    return new_model


# %%
snap_folder = '/home/emariott/deepmagic/output_data/snapshots'

snapshots = glob(f'{snap_folder}/SEDenseNet121_position_noclean_Gold_2019-02-25_01-37-25-*.h5')
snapshots_sort = sorted(snapshots)

# %%
print(snapshots_sort)
print(len(snapshots_sort))

# %%
snap_to_ensemble = snapshots_sort[-5:-1] + snapshots_sort[1:4]
print(snap_to_ensemble)

# %%
snap_to_ensemble_best = [snapshots_sort[-5]] + [snapshots_sort[1]] + [snapshots_sort[3]]
print(snap_to_ensemble)

# %%
snap_to_ensemble_best_more = [snapshots_sort[-5]] + [snapshots_sort[1]] + [snapshots_sort[3]] + [snapshots_sort[5]] + [
    snapshots_sort[-4]]
print(snap_to_ensemble_best_more)

# %%
model_list_sedense = []
model = SEDenseNet121_position_l2()
all_weights = []
for snap in tqdm(snap_to_ensemble_best_more):
    model.load_weights(snap)
    weghts_single_model = model.get_weights()
    all_weights.append(weghts_single_model)

# %%
swa_model_best_more = compute_SWA(all_weights, model)
print('saving the swa...')
# %%
swa_model.save_weights(
    '/home/emariott/software_magic/output_data/swa_models/SeDense121_position_from41_SWA_from10to19.h5')
print('Fatto')
# %%
