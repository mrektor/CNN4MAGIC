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
                         want_log_energy=True,
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

        if want_log_energy:
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
                                  machine='24cores',
                                  want_golden=False,
                                  want_energy=False, want_log_energy=True,
                                  want_position=False,
                                  want_label=False,
                                  clean=False,
                                  apply_log_to_raw=False,
                                  include_time=True
                                  ):
    # % Load df and complement Diffuse

    if clean:
        if machine == 'towerino':
            folder_diffuse = '/ssdraptor/magic_data/data_processed/diffuse_6_3punto5'
            filepath_df_diffuse = '/ssdraptor/magic_data/complement/diffuse_clean_6_3punto5_big_df.pkl'  # '/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_df.pkl'
            filepath_complement_diffuse = '/ssdraptor/magic_data/complement/diffuse_clean_6_3punto5_complement.pkl'  # '/ssdraptor/magic_data/complement/diffuse_clean_complement_complement.pkl' #'/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_complement.pkl'
        elif machine == '24cores':
            folder_diffuse = '/data/magic_data/clean_6_3punto5/montecarlo_diffuse/npy_dump'
            filepath_df_diffuse = '/data/magic_data/clean_6_3punto5/montecarlo_diffuse/big_df.pkl'
            filepath_complement_diffuse = '/data/magic_data/clean_10_5/diffuse_MC/diffuse_clean_10_5_complement.pkl'


    else:
        if machine == 'towerino':
            folder_point = '/ssdraptor/magic_data/data_processed/point_like'
            folder_diffuse = '/ssdraptor/magic_data/data_processed/diffuse'

            filepath_df_diffuse = '/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_df.pkl'
            filepath_complement_diffuse = '/home/emariott/deepmagic/data_interpolated/diffuse_complementary/diffuse_complement.pkl'

            filepath_df_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_df.pkl'
            filepath_complement_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_complement.pkl'

        elif machine == 'titanx':
            folder_point = '/home/emariott/point_like'
            folder_diffuse = '/home/emariott/diffuse'

            filepath_df_diffuse = '/home/emariott/complement/diffuse_df.pkl'
            filepath_complement_diffuse = '/home/emariott/complement/diffuse_complement.pkl'

            filepath_df_point = '/home/emariott/complement/point_df.pkl'
            filepath_complement_point = '/home/emariott/complement/point_complement.pkl'

    with open(filepath_df_diffuse, 'rb') as f:
        big_df_diffuse = pkl.load(f)

    with open(filepath_complement_diffuse, 'rb') as f:
        eventList_diffuse, labels_diffuse, energy_diffuse, position_diffuse = pkl.load(f)

    # % Load df and complement Point-Like
    if want_position or want_energy:
        # filepath_df_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_df.pkl'
        with open(filepath_df_point, 'rb') as f:
            big_df_point = pkl.load(f)

        # filepath_complement_point = '/home/emariott/deepmagic/data_interpolated/point_like_complementary/point_complement.pkl'
        with open(filepath_complement_point, 'rb') as f:
            _, labels_point, energy_point, position_point = pkl.load(f)

    if want_golden:
        # % Select the golden dataset
        golden_df_diffuse = big_df_diffuse[
            # (big_df_diffuse['impact_M1'] < 11000) &
            # (big_df_diffuse['impact_M2'] < 11000) &
            # (big_df_diffuse['impact_M1'] > 5000) &
            # (big_df_diffuse['impact_M2'] > 5000) &
            (big_df_diffuse['intensity_M1'] > 50) &
            (big_df_diffuse['intensity_M2'] > 50) &
            (big_df_diffuse['leakage2_pixel_M1'] < 0.2) &
            (big_df_diffuse['leakage2_pixel_M2'] < 0.2)
            ]

        golden_df_point = big_df_point[
            # (big_df_point['impact_M1'] < 11000) &
            # (big_df_point['impact_M2'] < 11000) &
            # (big_df_point['impact_M1'] > 5000) &
            # (big_df_point['impact_M2'] > 5000) &
            (big_df_point['intensity_M1'] > 50) &
            (big_df_point['intensity_M2'] > 50) &
            (big_df_point['leakage2_pixel_M1'] < 0.2) &
            (big_df_point['leakage2_pixel_M2'] < 0.2)
            ]

        ids_diffuse = golden_df_diffuse['ID'].values
        ids_point = golden_df_point['ID'].values
    else:
        ids_diffuse = big_df_diffuse['ID'].values
        if want_energy or want_position:
            ids_point = big_df_point['ID'].values

    if want_energy or want_position:
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
        if want_log_energy:
            energy_diffuse = {k: np.log10(v) for k, v in energy_diffuse.items()}  # Convert energies in log10
            energy_point = {k: np.log10(v) for k, v in energy_point.items()}

        # % Define the generators
        train_gn = MAGIC_Generator(list_IDs=partition['train'],
                                   labels=energy_diffuse,
                                   batch_size=batch_size,
                                   folder=folder_diffuse,
                                   energy=True,
                                   include_time=include_time,
                                   apply_log_to_raw=apply_log_to_raw
                                   )

        val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                                 labels=energy_diffuse,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_diffuse,
                                 energy=True,
                                 include_time=include_time,
                                 apply_log_to_raw=apply_log_to_raw
                                 )

        test_gn = MAGIC_Generator(list_IDs=partition['test'],
                                  labels=energy_point,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_point,
                                  energy=True,
                                  include_time=include_time,
                                  apply_log_to_raw=apply_log_to_raw
                                  )
        # %
        energy_vect = np.array([energy_point[event] for event in partition['test']])

        return train_gn, val_gn, test_gn, energy_vect
    # %
    if want_position:
        # % Define the generators
        train_gn = MAGIC_Generator(list_IDs=partition['train'],
                                   labels=position_diffuse,
                                   position=True,
                                   batch_size=batch_size,
                                   folder=folder_diffuse,
                                   apply_log_to_raw=apply_log_to_raw,
                                   include_time=include_time
                                   )

        val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                                 labels=position_diffuse,
                                 position=True,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_diffuse,
                                 apply_log_to_raw=apply_log_to_raw,
                                 include_time=include_time
                                 )

        test_gn = MAGIC_Generator(list_IDs=partition['test'],
                                  labels=position_point,
                                  position=True,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_point,
                                  apply_log_to_raw=apply_log_to_raw,
                                  include_time=include_time
                                  )

        position_vect = np.array([position_point[event] for event in partition['test']])
        return train_gn, val_gn, test_gn, position_vect

    if want_label:
        folder_global = '/data/magic_data/clean_10_5/all_npys/npy_dump'
        folder_point = '/data/magic_data/clean_10_5/point_MC/npy_dump'
        # folder_realdata = '/data/magic_data/clean_6_3punto5/cyn_1ES2037/npy_dump'
        filepath_complement_realdata = '/data/magic_data/clean_10_5/cyn_1ES2037/events_labels.pkl'
        with open(filepath_complement_realdata, 'rb') as f:
            eventList_realdata, labels_realdata = pkl.load(f)

        filepath_complement_diffuse_clean = '/data/magic_data/clean_10_5/diffuse_MC/diffuse_clean_10_5_complement.pkl'
        with open(filepath_complement_diffuse_clean, 'rb') as f:
            eventList_diffuse, labels_diffuse, _, _ = pkl.load(f)

        filepath_complement_point_clean = '/data/magic_data/clean_10_5/point_MC/point_clean_10_5_complement.pkl'
        with open(filepath_complement_point_clean, 'rb') as f:
            eventList_point, labels_point, _, _ = pkl.load(f)

        eventList_global = eventList_diffuse + eventList_realdata
        # eventList_global = glob.glob(f'{folder_global}/*.npy')
        labels_global = dict()
        labels_global.update(labels_diffuse)
        labels_global.update(labels_realdata)

        random.seed(42)
        random.shuffle(eventList_global)
        partition = dict()
        num_events = len(eventList_global)
        frac_tr_te = 0.70
        partition['train'] = eventList_global[:int(num_events * frac_tr_te)]
        partition['validation'] = eventList_global[int(num_events * frac_tr_te):]

        print(
            f'Training on {len(eventList_global[:int(num_events * frac_tr_te)])}\n Validating on {len(eventList_global[int(num_events * frac_tr_te):])}')

        train_gn = MAGIC_Generator(list_IDs=partition['train'],
                                   labels=labels_global,
                                   separation=True,
                                   batch_size=batch_size,
                                   folder=folder_global,
                                   apply_log_to_raw=apply_log_to_raw,
                                   include_time=include_time
                                   )

        val_gn = MAGIC_Generator(list_IDs=partition['validation'],
                                 labels=labels_global,
                                 separation=True,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 folder=folder_global,
                                 apply_log_to_raw=apply_log_to_raw,
                                 include_time=include_time
                                 )

        test_gn = MAGIC_Generator(list_IDs=eventList_point,
                                  labels=labels_point,
                                  separation=True,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  folder=folder_point,
                                  apply_log_to_raw=apply_log_to_raw,
                                  include_time=include_time
                                  )

        return train_gn, val_gn, test_gn
