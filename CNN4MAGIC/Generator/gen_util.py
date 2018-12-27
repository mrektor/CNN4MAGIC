import pickle as pkl

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


def load_data_generators(batch_size=400, want_energy=False, want_position=False, want_labels=False, want_test=False):
    # load IDs
    print('Loading labels...')
    filename = '/data2T/mariotti_data_2/MC_npy/complementary_dump_total.pkl'
    with open(filename, 'rb') as f:
        data, energy, labels, position = pkl.load(f)

    # %%
    print('Solving sponi...')
    data['train'] = clean_missing_data(data['train'], position)
    data['test'] = clean_missing_data(data['test'], position)
    data['validation'] = clean_missing_data(data['validation'], position)

    # %% Define the generators
    train_gn = MAGIC_Generator(list_IDs=data['train'],
                               labels=position,
                               position=True,
                               batch_size=batch_size,
                               folder='/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish'
                               )

    val_gn = MAGIC_Generator(list_IDs=data['validation'],
                             labels=position,
                             position=True,
                             batch_size=batch_size,
                             folder='/data2T/mariotti_data_2/MC_npy/finish_dump_MC/partial_dump_finish'
                             )

    if want_energy:
        energy = {k: np.log10(v) for k, v in energy.items()}  # Convert energies in log10
        return train_gn, val_gn, energy

    if want_labels:
        return train_gn, val_gn, labels

    if want_position:
        return train_gn, val_gn, position

    return train_gn, val_gn
