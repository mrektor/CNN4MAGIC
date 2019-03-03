import pickle

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import MobileNetV2_separation

# %%

with open('/data/magic_data/clean_6_3punto5/crab/events_labels.pkl', 'rb') as f:
    crabID, labels = pickle.load(f)

# %%
# Load the data
BATCH_SIZE = 128
crab_generator = MAGIC_Generator(list_IDs=crabID,
                                 labels=labels,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 folder='/data/magic_data/clean_6_3punto5/crab/npy_dump',
                                 include_time=False
                                 )
# %
model = MobileNetV2_separation(alpha=0.2, include_time=False)
weights_path = '/data/new_magic/output_data/snapshots/MobileNetV2_separation_PheOnly_clean6_3punto5_Gold_2019-02-27_17-10-41-Best.h5'
model.load_weights(weights_path)
y_pred_test = model.predict_generator(crab_generator, workers=24, verbose=1)
net_name = 'MobileNetV2_separation_alpha0.2_notime'
dump_name = f'output_data/reconstructions/separation_MC_{net_name}.pkl'
with open(dump_name, 'wb') as f:
    pickle.dump(y_pred_test, f)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(y_pred_test, bins=100, log=False)
plt.xlabel('Gammaness')
plt.ylabel('Counts')
plt.title(f'Gammaness of {y_pred_test.shape[0]} Triggered Crab events')
plt.savefig(f'output_data/pictures/gammaness_crab_{net_name}_nolog.png')
plt.close()

# %%
import numpy as np

gamma_like = y_pred_test > 0.5
print(np.sum(gamma_like))

# %%
from itertools import compress

gamma_like_id = list(compress(crabID, gamma_like))

# %%
from tqdm import tqdm

crab_folder = '/data/magic_data/clean_6_3punto5/crab/npy_dump'
for i in tqdm(range(len(gamma_like_id))):
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
    plt.savefig(f'/data/new_magic/output_data/pictures/crab_gamma_like/MV2_alpha0.2_notime/{event_id}.png')
    plt.close()
