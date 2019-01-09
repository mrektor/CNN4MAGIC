from keras.models import load_model
from CNN4MAGIC.CNN_Models.BigData.utils import plot_hist2D, plot_gaussian_error
from CNN4MAGIC.Generator.gen_util import load_data_generators

BATCH_SIZE = 475
train_gn, val_gn, test_gn, energy_te = load_data_generators(batch_size=BATCH_SIZE, want_energy=True)

net_name = 'MobileNetV2_energy-900kTrain'
filepath='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/checkpoints/MobileNetV2-alpha1-buonanno-energy-transfer.hdf5'
model = load_model(filepath)
#%
# import numpy as np
# sintetic = np.random.randn(100, 67, 68, 4)
# sintetic_label = np.random.randint(0, 1, 100)

#%
# model.fit(sintetic, sintetic_label, epochs=1)

#%
#%
print(len(energy_te))
# print()
#%
print(len(test_gn)*475)
#%
print('Making predictions on test set...')
y_pred = model.predict_generator(generator=test_gn, verbose=1, use_multiprocessing=True, workers=8)
y_test = energy_te

#%
print(y_pred[0:5])
print(y_test[0:5])
#%%
print(len(y_pred))
print(len(y_test))





##########################3

import glob
import pickle as pkl
import random

import numpy as np

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator


def clean_missing_data(data, labels):
    p = 0
    todelete = []
    for key in data:
        try:
            a = labels[key]
        except KeyError:
            todelete.append(key)
            p = p + 1
    print(f'solved {len(todelete)} of KeyErrors.')
    for key in todelete:
        data.remove(key)
    return data


print('Loading labels...')
filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total_2.pkl'
with open(filename, 'rb') as f:
    _, energy, labels, position = pkl.load(f)

eventList_total = glob.glob('/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish/*')
newlist = []
for event in eventList_total:
    newlist.append(event[66:-4])

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

want_energy=True
if want_energy:

    print('Solving sponi...')
    data = dict()
    data['test'] = clean_missing_data(partition['test'], energy)


    energy_log = {k: np.log10(v) for k, v in energy.items()}  # Convert energies in log10



    test_gn = MAGIC_Generator(list_IDs=data['test'],
                              shuffle=False,
                              labels=energy_log,
                              position=True,
                              batch_size=BATCH_SIZE,
                              folder='/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish'
                              )

    te_energy = [energy_log[event] for event in data['test']]


########################3


#%%
print(len(te_energy))
print(len(y_test))
print(len(y_pred))
print(len(data['test']))

#%% 475
import numpy as np
y_test = np.array(y_test)


#%%
print('Plotting stuff...')
plot_hist2D(y_test, y_pred,
            fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/',
            net_name=net_name,
            num_bins=100)

#%%
plot_gaussian_error(y_test, y_pred,
                    net_name=net_name + '_13bin',
                    num_bins=13,
                    fig_folder='/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/')
print('plotting metrics done')

#%%
import matplotlib.pyplot as plt

plt.figure()
plt.hist(y_pred, bins=100)
plt.savefig('/data/y_pred_MobileNetV2.png')

#%%
print(np.mean(y_test))
print(np.var(y_test))

print(np.mean(y_pred))
print(np.var(y_pred))
#%%
plt.hist(y_test)


# %% plot training
# import matplotlib.pyplot as plt
#
# fig_folder = '/data/mariotti_data/CNN4MAGIC/CNN_Models/BigData/pics/'
# # summarize history for loss
# plt.plot(result.history['loss'])
# plt.plot(result.history['val_loss'])
# plt.title('model loss MSE')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(fig_folder + net_name + '_loss_MSE.png')
# plt.show()
#
# plt.plot(result.history['mean_absolute_error'])
# plt.plot(result.history['val_mean_absolute_error'])
# plt.title('model loss MAE')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(fig_folder + net_name + '_loss_MAE.png')
# plt.show()
#
# plt.plot(result.history['mean_absolute_percentage_error'])
# plt.plot(result.history['val_mean_absolute_percentage_error'])
# plt.title('model loss MAE')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(fig_folder + net_name + '_loss_MAPE.png')
# plt.show()
#
# print('plotting training done')
