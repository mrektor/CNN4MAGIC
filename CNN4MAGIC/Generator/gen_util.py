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


def load_data_generators(batch_size=400, want_energy=False, want_position=False, want_labels=False, want_test=False,
                         folder_files='/data/magic_data/very_big_folder'):
    # load IDs
    print('Loading labels...')

    # SSH GPU
    # filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total_2.pkl'

    # SSH 24 CPU
    if want_labels:
        filename = '/data/magic_data/mc_root_labels.pkl'
        with open(filename, 'rb') as f:
            labels = pkl.load(f)
    else:
        filename = '/data/magic_data/MC_npy/complementary_dump_total_2.pkl'
        with open(filename, 'rb') as f:
            _, energy, labels, position = pkl.load(f)

    if want_labels:
        eventList_total = glob.glob('/data/magic_data/very_big_folder/*')
    else:
        eventList_total = glob.glob('/data/magic_data/very_big_folder/*corsika*')
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

    if want_energy:
        # %%
        print('Solving sponi...')
        data = dict()
        data['train'] = clean_missing_data(partition['train'], energy)
        data['test'] = clean_missing_data(partition['test'], energy)
        data['validation'] = clean_missing_data(partition['validation'], energy)
        train_points = len(partition['train'])
        val_points = len(partition['validation'])

        print(f'Training on {train_points} data points')
        print(f'Validating on {val_points} data points')

        energy = {k: np.log10(v) for k, v in energy.items()}  # Convert energies in log10

        # %% Define the generators
        train_gn = MAGIC_Generator(list_IDs=data['train'],
                                   labels=energy,
                                   batch_size=batch_size,
                                   folder=folder_files
                                   )

        val_gn = MAGIC_Generator(list_IDs=data['validation'],
                                 labels=energy,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_files
                                 )

        test_gn = MAGIC_Generator(list_IDs=data['test'],
                                  labels=energy,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_files
                                  )

        energy_vect = [energy[event] for event in data['test']]

        return train_gn, val_gn, test_gn, energy_vect

    if want_labels:
        # %%
        print('Solving sponi...')
        data = dict()
        data['train'] = clean_missing_data(partition['train'], labels)
        data['test'] = clean_missing_data(partition['test'], labels)
        data['validation'] = clean_missing_data(partition['validation'], labels)
        train_points = len(data['train'])
        val_points = len(data['validation'])

        print(f'Training on {train_points} data points')
        print(f'Validating on {val_points} data points')

        # %% Define the generators
        train_gn = MAGIC_Generator(list_IDs=data['train'],
                                   labels=labels,
                                   batch_size=batch_size,
                                   folder=folder_files
                                   )

        val_gn = MAGIC_Generator(list_IDs=data['validation'],
                                 labels=labels,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 folder=folder_files
                                 )
        test_gn = MAGIC_Generator(list_IDs=data['test'],
                                  labels=labels,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_files
                                  )

        labels_vect = [labels[event] for event in data['test']]

        return train_gn, val_gn, test_gn, labels_vect

    if want_position:
        # %%
        print('Solving sponi...')
        data = dict()
        data['train'] = clean_missing_data(partition['train'], position)
        data['test'] = clean_missing_data(partition['test'], position)
        data['validation'] = clean_missing_data(partition['validation'], position)
        train_points = len(data['train'])
        val_points = len(data['validation'])

        print(f'Training on {train_points} data points')
        print(f'Validating on {val_points} data points')

        # %% Define the generators
        train_gn = MAGIC_Generator(list_IDs=data['train'],
                                   labels=position,
                                   position=True,
                                   batch_size=batch_size,
                                   folder=folder_files
                                   )

        val_gn = MAGIC_Generator(list_IDs=data['validation'],
                                 labels=position,
                                 position=True,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_files
                                 )

        test_gn = MAGIC_Generator(list_IDs=data['test'],
                                  labels=position,
                                  position=True,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_files
                                  )

        position_vect = [position[event] for event in data['test']]

        return train_gn, val_gn, test_gn, position_vect
