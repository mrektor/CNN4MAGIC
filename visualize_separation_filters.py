print('Loading the Neural Network...')
# model = MobileNetV2_separation(alpha=0.2, include_time=False)
from CNN4MAGIC.Generator.models import dummy_cnn

model = dummy_cnn()
# %%
filename = '/data/new_magic/output_data/snapshots/dummy_cnn_4filter_1dense_nobias_2019-03-05_20-52-01-Best.h5'
model.load_weights(filename)
# %%
folder_fig = '/data/new_magic/view_filters'

# %%
import matplotlib.pyplot as plt


def plot_filer(filter_number):
    folder_fig = '/data/new_magic/view_filters'

    # filter_number = 0
    kernels = model.layers[1].get_weights()[0]
    single_kernel_0 = kernels[:, :, 0, filter_number]
    single_kernel_1 = kernels[:, :, 1, filter_number]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(single_kernel_0)
    plt.colorbar()
    plt.title(f'M1')

    plt.subplot(1, 2, 2)
    plt.imshow(single_kernel_1)
    plt.colorbar()
    plt.title('M2')

    plt.tight_layout()
    plt.savefig(f'{folder_fig}/kernel_{filter_number}.png')
    plt.close()


# %%
dense_values = model.layers[-1].get_weights()[0]

# %%
import numpy as np

# max_montecarlo = np.argmax(dense_values[0])
# max_real = np.argmin(dense_values[0])
for i in range(4):
    plot_filer(i)
# plot_filer(max_real)

# %%

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
kernels = model.layers[1].get_weights()[0]
vmin = np.min(kernels.flatten())
vmax = np.max(kernels.flatten())
for i in range(4):
    im = axes[0, i].imshow(kernels[:, :, 0, i], interpolation='nearest')
    axes[0, i].set_title(f'Kernel {i}, ch 0')

    axes[1, i].imshow(kernels[:, :, 1, i], interpolation='nearest')
    axes[1, i].set_title(f'Kernel {i}, ch 1')

    # divider4 = make_axes_locatable(axes[1, i])
    # cax4 = divider4.append_axes("right", size="20%", pad=0.05)
    # cbar4 = plt.colorbar(axes[1, i], cax=cax4)

plt.tight_layout()
plt.savefig(f'{folder_fig}/all_kernels_no_colorbar.png')
plt.close()

# %%
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
kernels, bias = model.layers[1].get_weights()
i = 0
for ax in axes:
    for subax in ax:
        subax.imshow(kernels[:, :, 0, i], interpolation='nearest')
        subax.set_title(f'Filter {i}')
        i = i + 1
plt.tight_layout()
plt.savefig(f'{folder_fig}/all_kernels.png')

# %%
plt.figure()
ind = np.arange(0, 4)
plt.bar(ind, dense_values.flatten())
plt.hlines(0, -0.5, 3.5, 'r')
plt.title('Dense Values')
plt.xticks([0, 1, 2, 3])
plt.xlabel('Feature Map')
plt.ylabel('Coefficient')
plt.savefig(f'{folder_fig}/dense.png')
plt.close()

# %%
from keras.models import Model

detected_feats = model.layers[2].output
input_layer = model.layers[0].input

intermediate_layer_model = Model(input_layer, detected_feats)
intermediate_layer_model.compile('sgd', 'mse')
# %%
from CNN4MAGIC.Generator.gen_util import load_generators_diffuse_point

BATCH_SIZE = 50
machine = '24cores'
train_gn, val_gn = load_generators_diffuse_point(
    batch_size=BATCH_SIZE,
    want_golden=False,
    want_label=True,
    machine=machine,
    clean=True,
    include_time=False)
# %%
batch = val_gn[0][0]
values = val_gn[0][1]
out_layer = intermediate_layer_model.predict_on_batch(batch)
pred = model.predict_on_batch(batch)
# %%
print(out_layer.shape)
el_idx = 0


def plot_event(el_idx, folder_fig):
    fig, axes = plt.subplots(1, 2)
    # kernels = model.layers[1].get_weights()[0]
    # vmin=np.min(kernels.flatten())
    # vmax=np.max(kernels.flatten())
    for i in range(1):
        axes[0].imshow(batch[el_idx, :, :, 0])
        axes[0].set_title('M1')
        axes[1].imshow(batch[el_idx, :, :, 1])
        axes[1].set_title('M2')

    plt.suptitle(f'Event {el_idx}: true label {values[el_idx]}')
    plt.tight_layout()
    plt.savefig(f'{folder_fig}/event_{el_idx}.png')
    plt.close()


# %%

def plot_output_conv_relu(el_idx, folder_fig):
    fig, axes = plt.subplots(2, 2)
    min_val = np.min(out_layer[el_idx, :, :, :])
    max_val = np.max(out_layer[el_idx, :, :, :])
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(out_layer[el_idx, :, :, i], cmap='inferno', vmin=min_val, vmax=max_val)
        ax.set_title(f'Output of kernel {i}')

    plt.suptitle(f'Event {el_idx}, true label = {values[el_idx]}, pred = {pred[el_idx][0]:.3f}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{folder_fig}/output_layer_{el_idx}.png')
    plt.close()


# %%
from tqdm import tqdm

new_folder_pic = '/data/new_magic/view_filters/experiment'
for i in tqdm(range(10)):
    plot_event(i, new_folder_pic)
    plot_output_conv_relu(i, new_folder_pic)

# %%
el_idx = 0
import matplotlib.patches as patches

# %%
i = 2
from numpy import unravel_index


def get_coord_max(el_idx, feature_map, return_feat_coord=False):
    idxs = unravel_index(out_layer[el_idx, :, :, feature_map].argmax(), out_layer[el_idx, :, :, feature_map].shape)
    # idxs_input = np.array(idxs)*2
    swapped = np.array([idxs[1], idxs[0]])
    if return_feat_coord:
        return swapped
    else:
        return swapped * 2


print(get_coord_max(1, 1))


# %%


def plot_event_featmap(el_idx, folder_fig, do_text=True):
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    filter_axes = axes[:, 1:]
    # %
    # kernels = model.layers[1].get_weights()[0]
    # vmin=np.min(kernels.flatten())
    # vmax=np.max(kernels.flatten())
    # for i in range(1):
    axes[0, 0].imshow(batch[el_idx, :, :, 0])
    axes[0, 0].set_title('M1')
    axes[1, 0].imshow(batch[el_idx, :, :, 1])
    axes[1, 0].set_title('M2')
    # Create a Rectangle patch
    rect_0 = patches.Rectangle(get_coord_max(el_idx, 0), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    rect_0_bis = patches.Rectangle(get_coord_max(el_idx, 0), 20, 20, linewidth=1, edgecolor='r', facecolor='none')

    rect_1 = patches.Rectangle(get_coord_max(el_idx, 1), 20, 20, linewidth=1, edgecolor='y', facecolor='none')
    rect_1_bis = patches.Rectangle(get_coord_max(el_idx, 1), 20, 20, linewidth=1, edgecolor='y', facecolor='none')

    rect_2 = patches.Rectangle(get_coord_max(el_idx, 2), 20, 20, linewidth=1, edgecolor='c', facecolor='none')
    rect_2_bis = patches.Rectangle(get_coord_max(el_idx, 2), 20, 20, linewidth=1, edgecolor='c', facecolor='none')

    rect_3 = patches.Rectangle(get_coord_max(el_idx, 3), 20, 20, linewidth=1, edgecolor='w', facecolor='none')
    rect_3_bis = patches.Rectangle(get_coord_max(el_idx, 3), 20, 20, linewidth=1, edgecolor='w', facecolor='none')

    # Add the patch to the Axes
    axes[1, 0].add_patch(rect_0)
    axes[0, 0].add_patch(rect_0_bis)
    # axes[1, 0].text(get_coord_max(el_idx, 0)[0],get_coord_max(el_idx, 0)[1], 'Filter 0', color='r')
    # axes[0, 0].text(get_coord_max(el_idx, 0)[0],get_coord_max(el_idx, 0)[1], 'Filter 0', color='r')

    axes[1, 0].add_patch(rect_1)
    axes[0, 0].add_patch(rect_1_bis)
    axes[1, 0].add_patch(rect_2)
    axes[0, 0].add_patch(rect_2_bis)
    axes[1, 0].add_patch(rect_3)
    axes[0, 0].add_patch(rect_3_bis)
    if do_text:
        axes[1, 0].text(get_coord_max(el_idx, 0)[0], get_coord_max(el_idx, 0)[1], 'Filter 0', color='r')
        axes[0, 0].text(get_coord_max(el_idx, 0)[0], get_coord_max(el_idx, 0)[1], 'Filter 0', color='r')

        axes[1, 0].text(get_coord_max(el_idx, 1)[0], get_coord_max(el_idx, 1)[1], 'Filter 1', color='y')
        axes[0, 0].text(get_coord_max(el_idx, 1)[0], get_coord_max(el_idx, 1)[1], 'Filter 1', color='y')

        axes[1, 0].text(get_coord_max(el_idx, 2)[0], get_coord_max(el_idx, 2)[1], 'Filter 2', color='c')
        axes[0, 0].text(get_coord_max(el_idx, 2)[0], get_coord_max(el_idx, 2)[1], 'Filter 2', color='c')

        axes[1, 0].text(get_coord_max(el_idx, 3)[0], get_coord_max(el_idx, 3)[1], 'Filter 3', color='w')
        axes[0, 0].text(get_coord_max(el_idx, 3)[0], get_coord_max(el_idx, 3)[1], 'Filter 3', color='w')

    # plt.suptitle(f'Event {el_idx}: true label {values[el_idx]}')

    min_val = np.min(out_layer[el_idx, :, :, :])
    max_val = np.max(out_layer[el_idx, :, :, :])
    for i, ax in enumerate(filter_axes.flatten()):
        ax.imshow(out_layer[el_idx, :, :, i], cmap='inferno', vmin=min_val, vmax=max_val)
        ax.plot(get_coord_max(el_idx, i, True)[0], get_coord_max(el_idx, i, True)[1], 'gx')
        ax.legend(['Maximum'])
        ax.set_title(f'Output of kernel {i}')

    plt.suptitle(f'Event {el_idx}, true label = {values[el_idx]}, pred = {pred[el_idx][0]:.3f}', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{folder_fig}/event_and_output_layer_{el_idx}.png')
    plt.close()


plot_event_featmap(el_idx, '/data/new_magic/view_filters/receptive_size')
# plt.tight_layout()
# plt.savefig(f'{folder_fig}/event_{el_idx}.png')
# plt.close()
# %%
from tqdm import tqdm

new_folder_pic = '/data/new_magic/view_filters/batch_validation_event_featmap'
for i in tqdm(range(BATCH_SIZE)):
    plot_event_featmap(i, new_folder_pic)
