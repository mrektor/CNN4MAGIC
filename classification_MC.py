import matplotlib

matplotlib.use('agg')

import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
# from CNN4MAGIC.Generator.training_util import get_telegram_callback
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_radam import RAdam
from tqdm import tqdm

from CNN4MAGIC.Generator.keras_generator import MAGIC_Generator
from CNN4MAGIC.Generator.models import efficientNet_B0_separation, efficientNet_B1_separation, \
    efficientNet_B2_separation, efficientNet_B3_separation, efficientNet_B4_separation

max_epochs = 60
batches = [512, 256, 256, 128, 64]
net_names = ['EfficientNet_B0_dropout06', 'EfficientNet_B1_dropout06', 'EfficientNet_B2', 'EfficientNet_B3', 'EfficientNet_B4']
models_fz = [efficientNet_B0_separation, efficientNet_B1_separation, efficientNet_B2_separation,
             efficientNet_B3_separation, efficientNet_B4_separation]
for net_name, single_model_fz, single_batch_size in zip(net_names, models_fz, batches):
    BATCH_SIZE = single_batch_size
    model = single_model_fz(include_time=True)
    model.compile(optimizer=RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(f'/data4T/CNN4MAGIC/results/MC_classification/computed_data/checkpoints/{net_name}.h5',
                                 save_weights_only=True)
    stop = EarlyStopping(patience=3, restore_best_weights=True)
    # tg = get_telegram_callback(net_name, machine='towerino')
    all_callbacks = [checkpoint, stop]
    want_golden = False


    def pickle_read(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


    def pickle_dump(filepath, object):
        with open(filepath, 'wb') as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


    # % Load complement
    folder_complement = '/ssdraptor/magic_data/classification_MC/complementary'
    diffuse_filenames_list, diffuse_labels, diffuse_energies, diffuse_positions = pickle_read(
        f'{folder_complement}/diffuse_complement.pkl')
    point_filenames_list, point_labels, point_energies, point_positions = pickle_read(
        f'{folder_complement}/point_complement.pkl')
    diffuse_df = pickle_read(f'{folder_complement}/diffuse_df.pkl')
    point_df = pickle_read(f'{folder_complement}/point_df.pkl')
    protons_complement = pickle_read(f'{folder_complement}/protons_noclean_big_df_ID_Labels.pkl')

    protons_labels = protons_complement.set_index('ID').to_dict()['label']

    # % Point to all files folder
    npy_folder = '/ssdraptor/magic_data/classification_MC/all_npy'

    # % Make Clean
    golden_df_diffuse = diffuse_df[
        (diffuse_df['intensity_M1'] > 50) &
        (diffuse_df['intensity_M2'] > 50) &
        (diffuse_df['leakage2_pixel_M1'] < 0.2) &
        (diffuse_df['leakage2_pixel_M2'] < 0.2)
        ]

    golden_df_point = point_df[
        (point_df['intensity_M1'] > 50) &
        (point_df['intensity_M2'] > 50) &
        (point_df['leakage2_pixel_M1'] < 0.2) &
        (point_df['leakage2_pixel_M2'] < 0.2)
        ]

    if want_golden:
        ids_diffuse = golden_df_diffuse['ID'].values
        ids_point = golden_df_point['ID'].values
    else:
        ids_diffuse = diffuse_df['ID'].values
        ids_point = point_df['ID'].values

    ids_protons = protons_complement['ID'].values
    # % tr-va-te split

    num_protons = len(ids_protons)
    ids_protons_tr = ids_protons[:int(num_protons * 0.7)]
    ids_protons_va = ids_protons[int(num_protons * 0.7):int(num_protons * 0.8)]
    ids_protons_test = ids_protons[int(num_protons * 0.8):]

    num_diffuse = len(ids_diffuse)
    ids_diffuse_tr = ids_diffuse[:int(num_diffuse * 0.40)]
    ids_diffuse_va = ids_diffuse[int(num_diffuse * 0.40):int(num_diffuse * 0.407)]
    ids_diffuse_te = ids_diffuse[int(num_diffuse * 0.40008):int(num_diffuse * 0.80)]

    num_point = len(ids_point)
    # ids_point_tr = ids_point[:int(num_point * 0.6)]
    ids_point_va = ids_point[:int(num_point * 0.005)]
    ids_point_te = ids_point[int(num_point * 0.2):int(num_point * 0.2)+len(ids_protons_test)]

    # % Define file list
    train_list_global = list(ids_diffuse_tr)
    for i in range(7):  # Oversampling the hadron class
        train_list_global = train_list_global + list(ids_protons_tr)

    validation_list_global = list(ids_protons_va) + list(ids_diffuse_va) + list(ids_point_va)
    test_list_global = list(ids_protons_test) + list(ids_diffuse_te) + list(ids_point_te)

    # %
    print(len(ids_protons_tr), len(ids_diffuse_tr))
    print(len(ids_protons_va), len(ids_diffuse_va), len(ids_point_va))
    print(len(ids_protons_test), len(ids_diffuse_te), len(ids_point_te))

    # % Define global label lookup dictionary
    global_lookup_labels = dict()
    global_lookup_labels.update(point_labels)
    global_lookup_labels.update(diffuse_labels)
    global_lookup_labels.update(protons_labels)
    print(len(global_lookup_labels))
    # % shuffle (might not be necessary)

    random.seed(42)
    random.shuffle(train_list_global)

    # % define generators
    # %#%%
    train_gn = MAGIC_Generator(list_IDs=train_list_global,
                               labels=global_lookup_labels,
                               separation=True,
                               batch_size=BATCH_SIZE,
                               folder=npy_folder,
                               apply_log_to_raw=False,
                               include_time=True
                               )

    val_gn = MAGIC_Generator(list_IDs=validation_list_global,
                             labels=global_lookup_labels,
                             separation=True,
                             shuffle=False,
                             batch_size=BATCH_SIZE,
                             folder=npy_folder,
                             apply_log_to_raw=False,
                             include_time=True
                             )

    # %%
    history = model.fit_generator(generator=train_gn,
                                  validation_data=val_gn,
                                  epochs=max_epochs,
                                  verbose=1,
                                  callbacks=all_callbacks,
                                  use_multiprocessing=False,
                                  workers=8)

    # %%
    hadron_test_gn = MAGIC_Generator(list_IDs=list(ids_protons_test),
                                     labels=global_lookup_labels,
                                     separation=True,
                                     shuffle=False,
                                     batch_size=BATCH_SIZE,
                                     folder=npy_folder,
                                     apply_log_to_raw=False,
                                     include_time=True
                                     )
    #
    diffuse_test_gn = MAGIC_Generator(list_IDs=list(ids_diffuse_te),
                                      labels=global_lookup_labels,
                                      separation=True,
                                      shuffle=False,
                                      batch_size=BATCH_SIZE,
                                      folder=npy_folder,
                                      apply_log_to_raw=False,
                                      include_time=True
                                      )
    #
    point_test_gn = MAGIC_Generator(list_IDs=list(ids_point_te),
                                    labels=global_lookup_labels,
                                    separation=True,
                                    shuffle=False,
                                    batch_size=BATCH_SIZE,
                                    folder=npy_folder,
                                    apply_log_to_raw=False,
                                    include_time=True
                                    )

    # %%
    prediction_hadron_test = model.predict_generator(hadron_test_gn, verbose=1, workers=8, max_queue_size=50)
    prediction_diffuse_test = model.predict_generator(diffuse_test_gn, verbose=1, workers=8, max_queue_size=50)
    prediction_point_test = model.predict_generator(point_test_gn, verbose=1, workers=8, max_queue_size=50)

    dump_folder = '/data4T/CNN4MAGIC/results/MC_classification/experiments/'

    if not os.path.exists(f'{dump_folder}/{net_name}/computed_data'):
        os.makedirs(f'{dump_folder}/{net_name}/computed_data')
    pickle_dump(f'{dump_folder}/{net_name}/computed_data/pred_hadr_te.pkl', prediction_hadron_test)
    pickle_dump(f'{dump_folder}/{net_name}/computed_data/pred_diff_te.pkl', prediction_diffuse_test)
    pickle_dump(f'{dump_folder}/{net_name}/computed_data/pred_point_te.pkl', prediction_point_test)
    pickle_dump(f'{dump_folder}/{net_name}/computed_data/history.pkl', history)
    model.save(f'{dump_folder}/{net_name}/computed_data/final_{net_name}.h5')
    # %
    # prediction_hadron_test = pickle_read(
    #     '/data4T/CNN4MAGIC/results/MC_classification/computed_data/pred_hadr_te.pkl')
    # prediction_diffuse_test = pickle_read(
    #     '/data4T/CNN4MAGIC/results/MC_classification/computed_data/pred_diff_te.pkl')
    # prediction_point_test = pickle_read(
    #     '/data4T/CNN4MAGIC/results/MC_classification/computed_data/pred_point_te.pkl')
    # %
    # %%
    if not os.path.exists(f'{dump_folder}/{net_name}/plots'):
        os.makedirs(f'{dump_folder}/{net_name}/plots')

    folder_pic = f'{dump_folder}/{net_name}/plots'
    plt.figure(figsize=(14, 8))
    plt.hist(prediction_hadron_test, label='Hadrons', bins=100, alpha=0.5, log=True)
    plt.hist(prediction_diffuse_test, label='Diffuse $\gamma$', bins=100, alpha=0.5, log=True)
    plt.hist(prediction_point_test, label='Point-Like $\gamma$', bins=100, alpha=0.5, log=True)
    plt.title('Distribution of Gammaness on Test Set (MC Classifcation)')
    plt.legend()
    plt.xlabel('Gammaness')
    plt.ylabel('Counts (Log scale)')
    plt.tight_layout()
    plt.savefig(f'{folder_pic}/gammaness_plot.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    plt.hist(-np.log(1 - prediction_hadron_test + 1e-11), label='Hadrons', bins=100, alpha=0.5, log=True)
    plt.hist(-np.log(1 - prediction_diffuse_test + 1e-11), label='Diffuse $\gamma$', bins=100, alpha=0.5, log=True)
    plt.hist(-np.log(1 - prediction_point_test + 1e-11), label='Point-Like $\gamma$', bins=100, alpha=0.5, log=True)
    plt.title('Distribution of Gammaness on Test Set (MC Classifcation)')
    plt.legend()
    plt.xlabel('-Log(1-Gammaness)')
    plt.ylabel('Counts (Log scale)')
    plt.tight_layout()
    plt.savefig(f'{folder_pic}/log_gammaness_plot.png')
    plt.close()


    def efficiency(prob_vector, num_cuts=100):
        total_number_events = len(prob_vector)
        eff_list = []
        gammaness_range = np.linspace(1e-9, 1 - 1e-9, num_cuts)
        for gammaness_cut in gammaness_range:
            num_selected = np.sum(prob_vector > gammaness_cut)
            eff_list.append(num_selected / total_number_events)
        return np.array(eff_list)


    point_eff = efficiency(prediction_point_test)
    hadron_eff = efficiency(prediction_hadron_test)
    q_factor = point_eff / np.sqrt(hadron_eff)

    plt.figure(figsize=(14, 8))
    plt.plot(np.linspace(1e-9, 1 - 1e-9, 100), q_factor, '-o')
    plt.ylabel('Q Factor')
    plt.xlabel('Gammaness')
    plt.title('Q Factor (MC classification)')
    plt.savefig(f'{folder_pic}/notime_q_factor.png')
    plt.close()


    def plot_misclassified(generator, predictions, gammaness_threshold=0.5, folder_misc=''):
        if not os.path.exists(f'{folder_pic}/{folder_misc}'):
            os.makedirs(f'{folder_pic}/{folder_misc}')

        folder_misc_complete = f'{folder_pic}/{folder_misc}'
        misclassified_bool = predictions[:, 0] < gammaness_threshold
        idx_misclassified = np.where(misclassified_bool)[0]

        batch_numbers = np.floor(idx_misclassified / BATCH_SIZE)
        idx_in_batches = np.mod(idx_misclassified, BATCH_SIZE)

        misclassified_events = [generator[int(batch_number)][0][idx_in_batch] for batch_number, idx_in_batch in
                                zip(batch_numbers, idx_in_batches)]

        gammaness = model.predict(np.array(misclassified_events))

        for misclassified_number, single_event in enumerate(tqdm(misclassified_events[:15])):
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            i = 0
            for ax in axes:
                ax[0].imshow(single_event[:, :, i])
                ax[1].imshow(single_event[:, :, i + 1])
                i += 2
            plt.suptitle(f'Gammaness: {gammaness[misclassified_number]}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{folder_misc_complete}/event_{misclassified_number}.png')
            plt.close()


    plot_misclassified(hadron_test_gn, prediction_hadron_test,
                       folder_misc='misclassified_hadrons')

    plot_misclassified(diffuse_test_gn, prediction_diffuse_test,
                       folder_misc='misclassified_diffuse')

    plot_misclassified(point_test_gn, prediction_point_test,
                       folder_misc='misclassified_point')
