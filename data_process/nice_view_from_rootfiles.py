from glob import glob

import matplotlib.pyplot as plt
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


# %%
m1, m2 = filenames[0]
print(m1, m2)
res = stereo_interp_from_root((m1, m2))

# %%
print(m1, m1[55:70])
# %%
files_m1_npy = glob(f'/ssdraptor/magic_data/data_processed/diffuse/*{m1[55:69]}*.npy')
print(len(files_m1_npy))
import numpy as np

npy_files = [np.load(file) for file in files_m1_npy]

# %%
print(res['M1_interp'].shape)
# %%
from keras.models import load_model

model = load_model('/home/emariott/deepmagic/CNN4MAGIC/Generator/checkpoints/MobileNetV2_4dense_position-big-2.hdf5')
# %%
single = npy_files[0]
print(single[np.newaxis].shape)
# %%
pos_pred = model.predict_on_batch(single[np.newaxis])


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
print(sorted(files_m1_npy))

# %%

i2 = res['M2_interp']
i1 = res['M1_interp']
print(i1.shape, i2.shape)
i12 = np.concatenate((i1, i2), axis=1)
print(i12.shape)

# %%
plt.figure()
plt.imshow(i12[20, 3, :, :], origin='lower')
plt.savefig(f'{folder_pic}/event_res_20_speriamo.png')
plt.close()

# %%
magic_plot_phe(m2, 0)
#%%
magic_plot_time(m1, idx_to_plot)
magic_plot_time(m2, idx_to_plot)

# %%
