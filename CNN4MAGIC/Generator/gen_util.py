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


def load_data_generators(batch_size=400,
                         want_energy=False, want_position=False, want_labels=False, want_point_test=False,
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


def load_point_generator(batch_size=400,
                         want_energy=False, want_position=False, want_labels=False,
                         folder_files='/home/emariott/deepmagic/data_interpolated/point_like'):
    complement_path = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_complement.pkl'
    with open(complement_path, 'rb') as f:
        event_list_total, labels_total, energy_total, position_total = pkl.load(f)

    random.seed(42)
    num_files = len(event_list_total)
    print(f'Number of files in folder: {num_files}')

    if want_energy:
        # %%

        log_energy = {k: np.log10(v) for k, v in energy_total.items()}  # Convert energies in log10

        # %% Define the generators
        generator = MAGIC_Generator(list_IDs=event_list_total,
                                    labels=log_energy,
                                    batch_size=batch_size,
                                    folder=folder_files,
                                    shuffle=False
                                    )

        energy_vect = [log_energy[event] for event in event_list_total]

        return generator, energy_vect

    if want_labels:
        # %% Define the generators
        generator = MAGIC_Generator(list_IDs=event_list_total,
                                    labels=labels_total,
                                    batch_size=batch_size,
                                    folder=folder_files,
                                    shuffle=False
                                    )

        labels_vect = [labels_total[event] for event in event_list_total]

        return generator, labels_vect

    if want_position:
        # %% Define the generators
        generator = MAGIC_Generator(list_IDs=event_list_total,
                                    labels=position_total,
                                    position=True,
                                    batch_size=batch_size,
                                    folder=folder_files,
                                    shuffle=False
                                    )

        position_vect = [position_total[event] for event in event_list_total]

        return generator, position_vect


def load_generators_diffuse_point(batch_size,
                                  want_golden=False,
                                  want_energy=False,
                                  want_position=False,
                                  folder_diffuse='/home/emariott/deepmagic/data_interpolated/diffuse',
                                  folder_point='/home/emariott/deepmagic/data_interpolated/point_like',
                                  ):
    # % Load df and complement Diffuse
    filepath_df_diffuse = '/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_df.pkl'
    with open(filepath_df_diffuse, 'rb') as f:
        big_df_diffuse = pkl.load(f)

    filepath_complement_diffuse = '/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_complement.pkl'
    with open(filepath_complement_diffuse, 'rb') as f:
        _, labels_diffuse, energy_diffuse, position_diffuse = pkl.load(f)

    # % Load df and complement Point-Like
    filepath_df_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_df.pkl'
    with open(filepath_df_point, 'rb') as f:
        big_df_point = pkl.load(f)

    filepath_complement_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_complement.pkl'
    with open(filepath_complement_point, 'rb') as f:
        _, labels_point, energy_point, position_point = pkl.load(f)

    if want_golden:
        # % Select the golden dataset
        golden_df_diffuse = big_df_diffuse[
            (big_df_diffuse['impact_M1'] < 11000) &
            (big_df_diffuse['impact_M2'] < 11000) &
            (big_df_diffuse['impact_M1'] > 5000) &
            (big_df_diffuse['impact_M2'] > 5000) &
            (big_df_diffuse['intensity_M1'] > 100) &
            (big_df_diffuse['intensity_M2'] > 100) &
            (big_df_diffuse['leakage2_pixel_M1'] < 0.2) &
            (big_df_diffuse['leakage2_pixel_M2'] < 0.2)
            ]

        golden_df_point = big_df_point[
            (big_df_point['impact_M1'] < 11000) &
            (big_df_point['impact_M2'] < 11000) &
            (big_df_point['impact_M1'] > 5000) &
            (big_df_point['impact_M2'] > 5000) &
            (big_df_point['intensity_M1'] > 100) &
            (big_df_point['intensity_M2'] > 100) &
            (big_df_point['leakage2_pixel_M1'] < 0.2) &
            (big_df_point['leakage2_pixel_M2'] < 0.2)
            ]

        ids_diffuse = golden_df_diffuse['ID'].values
        ids_point = golden_df_point['ID'].values
    else:
        ids_diffuse = big_df_diffuse['ID'].values
        ids_point = big_df_point['ID'].values

    partition = dict()
    frac_train = 0.70
    num_files = len(ids_diffuse)
    partition['train'] = ids_diffuse[:int(num_files * frac_train)]
    partition['validation'] = ids_diffuse[int(num_files * frac_train):]
    partition['test'] = ids_point
    print(
        f'Training on {int(num_files * frac_train)} Diffuse\n Validating on {num_files-int(num_files * frac_train)} Diffuse\nTesting on {len(ids_point)} Point-Like')
    # %
    if want_energy:
        energy_diffuse = {k: np.log10(v) for k, v in energy_diffuse.items()}  # Convert energies in log10

        # % Define the generators
        train_gn = MAGIC_Generator(list_IDs=partition['train'],
                                   labels=energy_diffuse,
                                   batch_size=batch_size,
                                   folder=folder_diffuse,
                                   energy=True
                                   )

        val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                                 labels=energy_diffuse,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_diffuse,
                                 energy=True
                                 )

        test_gn = MAGIC_Generator(list_IDs=partition['test'],
                                  labels=energy_point,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_point,
                                  energy=True
                                  )
        # %
        energy_vect = np.array([energy_point[event] for event in partition['test']])
        energy_log = np.log10(energy_vect)
        return train_gn, val_gn, test_gn, energy_log
    # %
    if want_position:
        # % Define the generators
        train_gn = MAGIC_Generator(list_IDs=partition['train'],
                                   labels=position_diffuse,
                                   position=True,
                                   batch_size=batch_size,
                                   folder=folder_diffuse
                                   )

        val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                                 labels=position_diffuse,
                                 position=True,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_diffuse
                                 )

        test_gn = MAGIC_Generator(list_IDs=partition['test'],
                                  labels=position_point,
                                  position=True,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_point
                                  )

        position_vect = np.array([position_point[event] for event in partition['test']])
        return train_gn, val_gn, test_gn, position_vect
