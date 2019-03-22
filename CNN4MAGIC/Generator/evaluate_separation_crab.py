import pickle
from glob import glob

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import MobileNetV2_separation

# import matplotlib
#
# matplotlib.use('TkAgg')
# %%
# with open('/data/magic_data/clean_6_3punto5/crab/events_labels.pkl', 'rb') as f:
#     crabID, labels = pickle.load(f)

crab_npy_path = glob('/home/emariott/magic_data/crab_clean10_5/npy_dump/*.npy')
# %%
with open('/home/emariott/magic_data/crab/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl',
          'rb') as f:
    big_df, evt_list = pickle.load(f)

# %%
labels = {ID: 0 for ID in evt_list}
evt_list_dot = [f'{ID}.' for ID in evt_list]
# %%
# Load the data
BATCH_SIZE = 128
crab_generator = MAGIC_Generator(list_IDs=evt_list,
                                 labels=labels,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/home/emariott/magic_data/crab_clean10_5/npy_dump',
                                 include_time=False)

# %%
import numpy as np
from tqdm import tqdm

absent = []

for ID in tqdm(evt_list):
    try:
        a = np.load(f'/home/emariott/magic_data/crab_clean10_5/npy_dump/{ID}.npy')
    except FileNotFoundError:
        absent.append(ID)

# %%
print(absent)
print(len(absent))

# %%
model = MobileNetV2_separation(alpha=1, include_time=False)
weights_path = 'output_data/snapshots/MobileNetV2_separation_10_5_2019-03-11_22-00-11-Best.h5'
model.load_weights(weights_path)
# %%
y_pred_test = model.predict_generator(crab_generator, workers=8, verbose=1, use_multiprocessing=True)
# %
# net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
# dump_name = f'output_data/reconstructions/crab_separation_{net_name}.pkl'
# with open(dump_name, 'wb') as f:
#     pickle.dump(y_pred_test, f)
# %%
net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
dump_name = f'output_data/reconstructions/crab_separation_{net_name}.pkl'
with open(dump_name, 'rb') as f:
    separation_gammaness = pickle.load(f)

print(separation_gammaness.shape)
# prova = crab_generator[1840]
# %%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(y_pred_test, bins=100, log=True)
plt.xlabel('Gammaness')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Triggered Crab events (Cleaning 10-5)')
plt.savefig(f'output_data/pictures/EHI_gammaness_crab_{net_name}_log.png')
plt.close()

# %%
import numpy as np

gamma_like = y_pred_test > 0.5
print(np.sum(gamma_like))
print(f'Accuracy alla buona: {1-(np.sum(gamma_like)/y_pred_test.shape[0])}')
# %%
from itertools import compress

gamma_like_id = list(compress(evt_list, gamma_like))

# %%
from tqdm import tqdm

crab_folder = '/home/emariott/magic_data/crab_clean10_5/npy_dump'
for i in tqdm(range(150)):
    event_id = gamma_like_id[i]
    gammaness_id = y_pred_test[gamma_like][i]
    a = np.load(f'{crab_folder}/{event_id}.npy')

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(a[:, :, 0])
    plt.colorbar()
    plt.title('Time 1')

    plt.subplot(2, 2, 2)
    plt.imshow(a[:, :, 1])
    plt.colorbar()

    plt.title('Phe 1')

    plt.subplot(2, 2, 3)
    plt.imshow(a[:, :, 2])
    plt.colorbar()
    plt.title('Time 2')

    plt.subplot(2, 2, 4)
    plt.imshow(a[:, :, 3])
    plt.colorbar()
    plt.title('Phe 2')

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Gammaness: {gammaness_id}')
    plt.savefig(f'/data/new_magic/output_data/pictures/ss433_gamma_like/hadroness_1e-6/{event_id}.png')
    plt.close()
