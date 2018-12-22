import glob
import pickle as pkl
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import *

# %
# load IDs
print('Loading labels...')
filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total.pkl'
with open(filename, 'rb') as f:
    data, energy, labels, position = pkl.load(f)

# %%
import numpy as np

energy = {k: np.log10(v) for k, v in energy.items()}  # Convert energies in log10

# %% puto coñazo da cambiare pronto
eventList_total = glob.glob('/data2T/mariotti_data_2/MC_npy/partial_dump_MC/*')
newlist = []
for event in tqdm(eventList_total):
    newlist.append(event[47:-4])

eventList_total = newlist
random.seed(42)
random.shuffle(eventList_total)
num_files = len(eventList_total)
partition = {}
partition['train'] = eventList_total[:int(num_files / 2)]
partition['validation'] = eventList_total[int(num_files / 2):int(num_files * 3 / 2)]
partition['test'] = eventList_total[int(num_files * 3 / 2):]


def clean_missing_data(data, labels):
    p = 0
    todelete = []
    for key in data:
        try:
            a = labels[key]
        except KeyError:
            todelete.append(key)
            p = p + 1
    print(f'{len(todelete)} of KeyErrors')
    for key in todelete:
        data.remove(key)
    return data


partition['train'] = clean_missing_data(partition['train'], position)
partition['test'] = clean_missing_data(partition['test'], position)
partition['validation'] = clean_missing_data(partition['validation'], position)

# %%
# %% Define the generators
BATCH_SIZE = 64
train_gn = MAGIC_Generator(list_IDs=partition['train'],
                           labels=position,
                           position=True,
                           batch_size=BATCH_SIZE,
                           folder='/data2T/mariotti_data_2/MC_npy/partial_dump_MC'
                           )

val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                         labels=position,
                         position=True,
                         batch_size=BATCH_SIZE,
                         folder='/data2T/mariotti_data_2/MC_npy/partial_dump_MC'
                         )

# %% Load Model
print('Loading the Neural Network...')
model = MobileNetV2_slim_position()
model.compile(optimizer='sgd', loss='mse')
model.summary()

# %% Train
EPOCHS = 5

net_name = 'MobileNetV2_slim_energy_fitgen'
path = '/data/mariotti_data/CNN4MAGIC/CNN_Models/SeparationStereo/checkpoints/' + net_name
check = ModelCheckpoint(filepath=path, save_best_only=True)
clr = OneCycleLR(max_lr=0.0008,
                 num_epochs=EPOCHS,
                 num_samples=len(train_gn),
                 batch_size=BATCH_SIZE)
stop = EarlyStopping(patience=2)

result = model.fit_generator(generator=train_gn,
                             validation_data=val_gn,
                             epochs=EPOCHS,
                             verbose=1,
                             callbacks=[check, clr, stop],
                             use_multiprocessing=True,
                             workers=16
                             )
# %%
len(labels)
