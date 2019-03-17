import glob
import pickle

import numpy as np

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point


def pkl_load(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


# %%

BATCH_SIZE = 128
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    include_time=True,
    clean=False)

# %%
energy_reco_filepath = glob.glob('/home/emariott/deepmagic/output_data/reconstructions/*energy*.pkl')

# %%
networks = [path[53:-4] for path in energy_reco_filepath]
print(networks)

# %%
appello = {net: pkl_load(net_path) for net_path, net in zip(energy_reco_filepath, networks)}

# %%
print(appello)

# %%
lengths = [len(pred) for pred in appello.values()]
print(lengths)


# %%
def compute_loss(y_pred):
    # print(len(y_pred), len(energy))
    energy_limato = energy.flatten()[:len(y_pred)]
    mse = np.sum((energy_limato - y_pred.flatten()) ** 2)
    return mse


a = compute_loss(appello['energy_energy_skrr_30_best'])
print(a)
# %%
losses_dict = {net: compute_loss(appello[net]) for net in networks}
# %%
losses_list = [compute_loss(appello[net]) for net in networks]
# %%
import pandas as pd

# %%
df = pd.DataFrame(losses, index=losses.keys())
print(df)
# %%
fold_fig = '/home/emariott/deepmagic/output_data/pictures/pagelle'
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
plt.barh(networks, losses_list, log=False)
plt.tight_layout(h_pad=10)
plt.savefig(f'{fold_fig}/test.png')
