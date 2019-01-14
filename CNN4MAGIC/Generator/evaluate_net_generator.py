import numpy as np
from keras.models import load_model

from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 641
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_labels=True)

# %
print(len(energy_te))
print(len(test_gn) * BATCH_SIZE)
# %
labels = np.array(energy_te)

# %

net_name = 'MobileNetV2-separation-big'
filepath = '/data/code/CNN4MAGIC/Generator/checkpoints/MobileNetV2-separation-big.hdf5'
model = load_model(filepath)

# %
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=24)

#%%
print(y_pred[0:30].flatten())
print(labels[0:30].flatten())
print(y_pred.shape, labels.shape)

y_true = labels

# %%
from CNN4MAGIC.CNN_Models.SeparationStereo.utils import plot_classification_merit_metrics

# %%

hadrons = y_pred[y_true == 0]
gammas = y_pred[y_true == 1]
# sns.set()
bins = 85
plt.figure()
plt.hist(hadrons, bins=bins, log=True, histtype='step', fill=True, alpha=0.5)
plt.hist(gammas, bins=bins, log=True, histtype='step', fill=True, alpha=0.5)
plt.xlim([0, 1])
plt.legend(['Hadrons', 'Gammas'])
plt.title(net_name)
plt.xlabel('Gammaness')
plt.savefig('/data/code/CNN4MAGIC/Generator/separation_generator_pic/gammaness_' + net_name + '.png')
plt.savefig('/data/code/CNN4MAGIC/Generator/separation_generator_pic/gammaness_' + net_name + '.eps')
plt.show()

# %%
print('Plotting Merit Figures...')
plot_classification_merit_metrics(y_pred, y_true, net_name=net_name,
                                  fig_folder='/data/code/CNN4MAGIC/Generator/separation_generator_pic')
# %%
# Plot stuff
print('Plotting gammaness...')
plot_gammaness(y_pred, y_true, net_name=net_name,
               fig_folder='/data/code/CNN4MAGIC/Generator/separation_generator_pic')

# %%
print('Plotting confusion matrix...')
plot_confusion_matrix(y_pred, y_true, ['Hadrons', 'Gammas'], net_name=net_name,
                      fig_folder='/data/code/CNN4MAGIC/Generator/separation_generator_pic')

print('Plotting Merit Figures...')
plot_classification_merit_metrics(y_pred, y_true, net_name=net_name,
                                  fig_folder='/data/code/CNN4MAGIC/Generator/separation_generator_pic')
print('All done')

# %%
eventList_total = glob.glob('/data/magic_data/very_big_folder/*')
newlist = []
for event in eventList_total:
    newlist.append(event[33:-4])

eventList_total = newlist
random.seed(42)
random.shuffle(eventList_total)
num_files = len(eventList_total)
print(f'Number of files in folder: {num_files}')
partition = dict()
frac_train = 0.67
frac_val = 0.10
partition['train'] = eventList_total[:int(num_files * frac_train)]
partition['validation'] = eventList_total[int(num_files * frac_train):int(num_files * (frac_train + frac_val))]
partition['test'] = eventList_total[int(num_files * (frac_train + frac_val)):]
# %%

event_list_test = partition['test']

# %%
import pickle as pkl

filename = '/data/magic_data/mc_root_labels.pkl'
with open(filename, 'rb') as f:
    labels_dict = pkl.load(f)

# %%
predicted_as_gamma = y_pred > 0.5
is_hadron = y_true == 0

misclassified_hadrons_mask = np.logical_and(predicted_as_gamma.flatten(), is_hadron)

# %%
y_pred_flat = y_pred.flatten()
misclassified_probability = y_pred_flat[misclassified_hadrons_mask]
print(misclassified_probability)

# %%
mezzSu = y_pred_flat < 0.9
mezzGiu = y_pred_flat > 0.1

ue = np.logical_and(mezzGiu, mezzSu)
oi = y_pred_flat[ue]
print(oi.shape)

# %%
from itertools import compress

misclassified_hadrons_labels = list(compress(event_list_test, misclassified_hadrons_mask))

folder_file = '/data/magic_data/very_big_folder/'
data_misclassified_hadrons = np.array([np.load(folder_file + label + '.npy') for label in misclassified_hadrons_labels])

print(data_misclassified_hadrons.shape)

# %%
fig_folder = '/data/code/CNN4MAGIC/Generator/separation_generator_pic/'
num_events = data_misclassified_hadrons.shape[0]
fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))

for i in range(num_events):
    axes[i, 0].imshow(data_misclassified_hadrons[i, :, :, 0])  # TIME
    axes[i, 0].set_title('M1 Time')
    axes[i, 0].set_ylabel('Gammaness: ' + str(misclassified_probability[i]))

    axes[i, 1].imshow(data_misclassified_hadrons[i, :, :, 1])  # PHE
    axes[i, 1].set_title('M1 PHE')

    axes[i, 2].imshow(data_misclassified_hadrons[i, :, :, 2])  # TIME
    axes[i, 2].set_title('M2 Time')

    axes[i, 3].imshow(data_misclassified_hadrons[i, :, :, 3])  # PHE
    axes[i, 3].set_title('M2 PHE')

fig.suptitle('Hadrons misclassified as Gammas')
plt.tight_layout()
plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.png')
plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.pdf')
plt.show()

# %%


# %%
predicted_as_hadron = y_pred < 0.5
is_gamma = y_true == 1

misclassified_gamma_mask = np.logical_and(predicted_as_hadron.flatten(), is_gamma)

# %%
y_pred_flat = y_pred.flatten()
misclassified_gamma_probability = y_pred_flat[misclassified_gamma_mask]

# %%
from itertools import compress

misclassified_gamma_labels = list(compress(event_list_test, misclassified_gamma_mask))

folder_file = '/data/magic_data/very_big_folder/'
data_misclassified_gammas = np.array([np.load(folder_file + label + '.npy') for label in misclassified_gamma_labels])

# %%
print(data_misclassified_gammas.shape)
# %%

fig_folder = '/data/code/CNN4MAGIC/Generator/separation_generator_pic/'
num_events = 30
fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))

for i, idx in enumerate(range(30, 60)):
    axes[i, 0].imshow(data_misclassified_gammas[idx, :, :, 0])  # TIME
    axes[i, 0].set_title('M1 Time')
    axes[i, 0].set_ylabel('Gammaness: ' + str(misclassified_gamma_probability[idx]))

    axes[i, 1].imshow(data_misclassified_gammas[idx, :, :, 1])  # PHE
    axes[i, 1].set_title('M1 PHE')

    axes[i, 2].imshow(data_misclassified_gammas[idx, :, :, 2])  # TIME
    axes[i, 2].set_title('M2 Time')

    axes[i, 3].imshow(data_misclassified_gammas[idx, :, :, 3])  # PHE
    axes[i, 3].set_title('M2 PHE')

fig.suptitle('Gammas misclassified as Hadrons')
plt.tight_layout()
plt.savefig(fig_folder + net_name + 'MisclassifiedGammas0-30.png')
plt.savefig(fig_folder + net_name + 'MisclassifiedGammas0-30.pdf')
plt.show()

# %%
print(misclassified_gamma_probability)


# %%
def plot_misclassified_hadrons(m1_te, m2_te, y_pred_h, num_events=10, net_name='',
                               fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/pics/'):
    misclassified_hadrons_mask = y_pred_h > 0.5
    misclassified_probability = y_pred_h[misclassified_hadrons_mask.flatten()]
    misclassified_hadrons_M1 = m1_te[misclassified_hadrons_mask.flatten()]
    misclassified_hadrons_M2 = m2_te[misclassified_hadrons_mask.flatten()]

    print(f'there are {misclassified_hadrons_M1.shape[0]} misclassified hadrons')
    if num_events > misclassified_hadrons_M1.shape[0]:
        num_events = misclassified_hadrons_M1.shape[0]
    fig, axes = plt.subplots(num_events, 4, figsize=(15, num_events * 3))

    indexes = [i for i in range(misclassified_hadrons_M1.shape[0])]
    random.shuffle(indexes)
    for i, idx in enumerate(indexes[:num_events]):
        axes[i, 0].imshow(misclassified_hadrons_M1[idx, :, :, 0])  # TIME
        axes[i, 0].set_title('M1 Time')
        axes[i, 0].set_ylabel('Gammaness: ' + str(misclassified_probability[idx]))

        axes[i, 1].imshow(misclassified_hadrons_M1[idx, :, :, 1])  # PHE
        axes[i, 1].set_title('M1 PHE')

        axes[i, 2].imshow(misclassified_hadrons_M2[idx, :, :, 0])  # TIME
        axes[i, 2].set_title('M2 Time')

        axes[i, 3].imshow(misclassified_hadrons_M2[idx, :, :, 1])  # PHE
        axes[i, 3].set_title('M2 PHE')

    fig.suptitle('Hadrons misclassified as Gammas')
    plt.tight_layout()
    plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.png')
    plt.savefig(fig_folder + net_name + 'MisclassifiedHadrons.pdf')
    plt.show()
