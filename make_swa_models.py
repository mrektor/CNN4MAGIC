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

snap_list = glob(
    '/home/emariott/software_magic/output_data/snapshots/SEDenseNet121_position_l2_fromEpoch41_2019-03-07_17-31-27-1*.h5')
print(len(snap_list))
snap_sort = sorted(snap_list)
print(snap_sort)

# %%
model_list_sedense = []
model = SEDenseNet121_position_l2()
all_weights = []
for snap in tqdm(snap_sort):
    model.load_weights(snap)
    weghts_single_model = model.get_weights()
    all_weights.append(weghts_single_model)

# %%
swa_model = compute_SWA(all_weights, model)
print('saving the swa...')
# %%
swa_model.save_weights(
    '/home/emariott/software_magic/output_data/swa_models/SeDense121_position_from41_SWA_from10to19.h5')
print('Fatto')
# %%
