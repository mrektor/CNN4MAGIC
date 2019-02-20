from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

from data_process.position_stereo_visua import read_from_root, stereo_interp_from_root


# %%
def get_pair_match(a, b):
    result = []
    for i in a:
        for j in b:
            if i[-28:-5] == j[-28:-5]:  # -28:-5 for MC. -42:-6 for ROOT
                result.append((i, j))
    return result


point_files = glob('/home/emariott/deepmagic/data_root/mc/point_like/*.root')
diffuse_files_M1 = glob('/home/emariott/deepmagic/data_root/mc/diffuse/*M1*.root')
diffuse_files_M2 = glob('/home/emariott/deepmagic/data_root/mc/diffuse/*M2*.root')

filenames = get_pair_match(diffuse_files_M1, diffuse_files_M2)
print(len(filenames))

# %%
folder_pic = '/home/emariott/deepmagic/data_process/visualisation_pics'
df, phe, time = read_from_root(point_files[0])


def magic_plot_phe(filename, idx):
    df, phe, time = read_from_root(filename)
    phe_to_view = phe.iloc[idx, :1039]
    camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')
    plt.figure()
    display = CameraDisplay(camera_MAGIC)

    display.image = phe_to_view
    display.add_colorbar()
    plt.title(f'MAGIC cam Phe signal of event {idx}')

    plt.savefig(f'{folder_pic}/magic_camera_{filename[-30:-28]}_{idx}_phe.png')
    plt.close()


def magic_plot_time(filename, idx):
    df, phe, time = read_from_root(filename)
    time_to_view = time.iloc[idx, :1039]
    camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')
    plt.figure()
    display = CameraDisplay(camera_MAGIC)

    display.image = time_to_view
    display.add_colorbar()
    plt.title(f'MAGIC cam Time signal of event {idx}')

    plt.savefig(f'{folder_pic}/magic_camera_{filename[-30:-28]}_{idx}_time.png')
    plt.close()


# %%
m1, m2 = filenames[0]
print(m1, m2)
res = stereo_interp_from_root((m1, m2))

# %%

i2 = res['M2_interp']
i1 = res['M1_interp']
print(i1.shape, i2.shape)
i12 = []
for idx in range(i1.shape[0]):
    evt1 = i1[idx]
    evt2 = i2[idx]

    # print(evt1.shape, evt2.shape)
    b = np.zeros((67, 68, 4))
    b[:, :, 0] = evt1[0]
    b[:, :, 1] = evt1[1]
    b[:, :, 2] = evt2[0]
    b[:, :, 3] = evt2[1]
    i12.append(b)

i12_vect = np.array(i12)
print(i12_vect.shape)
# %%
from keras.models import load_model

model = load_model('/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5')
pos_pred = model.predict_on_batch(i12_vect)

# %%
# pos_true = np.concatenate([res['src_X1'], res['src_Y1']])
posx_true = res['src_X1'].values
posy_true = res['src_Y1'].values

posx_true_deg = posx_true * 0.00337
posy_true_deg = posy_true * 0.00337

pos_pred_deg = pos_pred * 0.00337

theta2 = np.power(pos_pred_deg[:, 0] - posx_true_deg, 2) + np.power((pos_pred_deg[:, 1] - posy_true_deg), 2)
print(theta2.shape)

quali_theta_2_piccoli = theta2 < 0.01
theta2_piccoli = theta2[theta2 < 0.01]

secondo_bin_theta = (theta2 > 0.01) & (theta2 < 0.04)


# %%
df, phe, time = read_from_root(m1)


# %%
def magic_plot_phe_pos(idx, pos_true_x, pos_true_y, pos_reco_x, pos_reco_y, highlight=False):
    phe_to_view = phe.iloc[idx, :1039]
    camera_MAGIC = CameraGeometry.from_name('MAGICCamMars')
    plt.figure()
    display = CameraDisplay(camera_MAGIC)

    display.image = phe_to_view
    display.add_colorbar()
    plt.plot(pos_true_x * 0.001, pos_true_y * 0.001, 'xr', markersize=15)
    plt.plot(pos_reco_x * 0.001, pos_reco_y * 0.001, 'xw', markersize=15)

    if highlight:
        display.highlight_pixels([i for i in range(len(phe_to_view.values))])
    plt.title(f'MAGIC cam Phe signal of event {idx}')

    plt.savefig(f'/home/emariott/deepmagic/data_process/visualisation_pics/pos_04_highlight/magic_camera_{idx}_phe.png')
    plt.close()


# %%
def get_index_from_bool(bool_idx):
    a = np.array([i for i in range(len(bool_idx))])[bool_idx]
    return a


# %%
t001 = get_index_from_bool(quali_theta_2_piccoli)
t004 = get_index_from_bool(secondo_bin_theta)
from tqdm import tqdm

for idx in tqdm(t004):
    magic_plot_phe_pos(idx, posx_true[idx], posy_true[idx], pos_pred[idx, 0], pos_pred[idx, 1], highlight=True)


# %%

def plot_interp(idx, plot_position=False):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    ph_m1 = axs[0, 0].imshow(res['M1_interp'][idx, 1, :, :], origin='lower')
    if plot_position:
        axs[0, 0].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=18)
    axs[0, 0].set_title('M1, Phe')
    # fig.colorbar(ph_m1, axs[0, 0])

    axs[0, 1].imshow(res['M1_interp'][idx, 0, :, :], origin='lower')
    if plot_position:
        axs[0, 1].plot(res['pos_interp1'][idx][0], res['pos_interp1'][idx][1], 'xr', markersize=18)
    axs[0, 1].set_title('M1, Time')

    axs[1, 0].imshow(res['M2_interp'][idx, 1, :, :], origin='lower')
    if plot_position:
        axs[1, 0].plot(res['pos_interp2'][idx][0], res['pos_interp2'][idx][1], 'xr', markersize=18)
    axs[1, 0].set_title('M2, Phe')

    axs[1, 1].imshow(res['M2_interp'][idx, 0, :, :], origin='lower')
    if plot_position:
        axs[1, 1].plot(res['pos_interp2'][idx][0], res['pos_interp2'][idx][1], 'xr', markersize=18)
    axs[1, 1].set_title('M2, Time')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Event ' + str(idx) + ' Energy = ' + str(res['energy'][idx]))
    plt.savefig(f'{folder_pic}/inerp_evt_test_{idx}.png')
    plt.show()


# plot_interp(2)

# %%
idx_to_plot = 47
plot_interp(idx_to_plot)
# %%
idx_to_plot = 0
magic_plot_phe(m1, idx_to_plot)

# %%
plt.figure()
plt.imshow(single[:, :, 3], origin='lower')
plt.savefig(f'{folder_pic}/event_0_speriamo.png')
plt.close()

# %%
print(res.keys())
# %%
print(res['corsika_event_number_1'])
# print(sorted(files_m1_npy))


# %%
plt.figure()
plt.imshow(i12[20, 3, :, :], origin='lower')
plt.savefig(f'{folder_pic}/event_res_20_speriamo.png')
plt.close()

# %%
magic_plot_phe(m2, 0)
# %%
magic_plot_time(m1, idx_to_plot)
magic_plot_time(m2, idx_to_plot)

# %%
