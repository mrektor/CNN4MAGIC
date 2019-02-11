import pickle

from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

pred_path = '/home/emariott/deepmagic/CNN4MAGIC/Generator/position_predictions/pos_predMobileNetV2_4dense_position-big-2.pkl'

with open(pred_path, 'rb') as f:
    y_pred = pickle.load(f)
# %%
# Load the data
BATCH_SIZE = 128
train_gn, val_gn, test_gn, positions = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                     want_golden=False,
                                                                     want_position=True)
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(batch_size=BATCH_SIZE,
                                                                  want_energy=True,
                                                                  want_golden=False)
# %%
print(y_pred.shape)
print(positions.shape)
print(energy.shape)

# %%
from CNN4MAGIC.CNN_Models.BigData.utils import plot_angular_resolution

positions_limato = positions[:y_pred.shape[0]]
energy_limato = energy[:y_pred.shape[0]]

plot_angular_resolution(positions_limato, y_pred, energy_limato, net_name='aaaah',
                        fig_folder='/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic')

# %%
fig_folder = '/home/emariott/deepmagic/CNN4MAGIC/Generator/position_pic'
import matplotlib.pyplot as plt

from data_process.InterpolateMagic import InterpolateMagic

interpolator = InterpolateMagic(15)

pred_pos = interpolator.interp_pos(y_pred[0, :])
true_pos = interpolator.interp_pos(positions_limato[0, :])

# %%
[evt, energy_evt] = test_gn[0]
from tqdm import tqdm

evt_idx = 0
for evt_idx in tqdm(range(100)):
    fig, axes = plt.subplots(2, 2)
    pred_pos = interpolator.interp_pos(y_pred[evt_idx, :])
    true_pos = interpolator.interp_pos(positions_limato[evt_idx, :])
    single_evt = evt[evt_idx, :, :, :]

    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(evt[evt_idx, :, :, idx])
        ax.plot([true_pos[0]], [true_pos[1]], 'xr')
        ax.plot([pred_pos[0]], [pred_pos[1]], 'om')
    plt.suptitle(f'Energy: {energy_evt[evt_idx]:.2f}')

    plt.savefig(f'{fig_folder}/evt{evt_idx}.png')
    plt.close()
# %%
print(evt.shape)
