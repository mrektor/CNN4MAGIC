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

crab_npy_path = glob('/ssdraptor/magic_data/crab/crab_data/crab_npy/*.npy')
# %%
with open('/ssdraptor/magic_data/crab/crab_data/crab_position_dataframe/big_df_complement_position_interpolated_nan.pkl',
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
                                 folder='/ssdraptor/magic_data/crab/crab_data/crab_npy',
                                 include_time=True)

# %%
import numpy as np
from tqdm import tqdm

absent = []

for ID in tqdm(evt_list[:10]):
    try:
        a = np.load(f'/ssdraptor/magic_data/crab/crab_data/crab_npy/{ID}.npy')
    except FileNotFoundError:
        absent.append(ID)

# %%
print(len(absent))
#%%
print(absent)

# %%
# model = MobileNetV2_separation(alpha=1, include_time=False)
# weights_path = 'output_data/snapshots/MobileNetV2_separation_10_5_2019-03-11_22-00-11-Best.h5'
# model.load_weights(weights_path)
from keras.models import load_model
model = load_model('/data4T/CNN4MAGIC/results/MC_classification/computed_data/one-epoch-MV2.h5')

# %%
y_pred_test = model.predict_generator(crab_generator, verbose=1)
# %%
net_name = 'one-epoch-MV2'
dump_name = f'/data4T/CNN4MAGIC/results/MC_classification/computed_data/crab_separation_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)
# %%
# net_name = 'MobileNetV2_separation_10_5_notime_alpha1'
# dump_name = f'output_data/reconstructions/crab_separation_{net_name}.pkl'
# with open(dump_name, 'rb') as f:
#     separation_gammaness = pickle.load(f)
#
# print(separation_gammaness.shape)
# prova = crab_generator[1840]
# %%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(y_pred_test, bins=100, log=True)
plt.xlabel('Gammaness')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Triggered Crab events')
plt.savefig(f'/data4T/CNN4MAGIC/results/MC_classification/plots/crab__gammaness_{net_name}_log.png')
plt.close()

#%%
plt.figure()
plt.hist(-np.log(1-y_pred_test+1e-10), bins=100, log=True)
plt.xlabel('-Log (1-Gammaness)')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Triggered Crab events')
plt.savefig(f'/data4T/CNN4MAGIC/results/MC_classification/plots/log_gammaness_{net_name}_log.png')
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
#%%%%% To delete
a_batch = diffuse_test_gn[0][0]
print(a_batch.shape)
#%%
from tqdm import tqdm

crab_folder = '/ssdraptor/magic_data/crab/crab_data/crab_npy'
for i in tqdm(range(10)):
    event_id = gamma_like_id[i]
    gammaness_id = y_pred_test[gamma_like][i]
    # a = np.load(f'{crab_folder}/{event_id}.npy')
    a = a_batch[i]
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
    plt.savefig(f'/data4T/CNN4MAGIC/results/MC_classification/plots/some_gammas/{i}.png')
    plt.close()
