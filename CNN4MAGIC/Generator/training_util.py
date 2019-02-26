import pickle
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from ..CNN_Models.BigData.clr import OneCycleLR
from ..CNN_Models.BigData.snapshot import SnapshotCallbackBuilder
from ..Other_utilities.dl_bot import DLBot
from ..Other_utilities.telegram_bot_callback import TelegramBotCallback


def get_telegram_callback(net_name='', machine='towerino'):
    if machine == 'towerino':
        telegram_token = '705094604:AAG8zNsLbcpExd_Ezhhw5TcHmgnZ---5PqM'  # replace TOKEN with your bot's token
    elif machine == '24cores':
        telegram_token = '645822793:AAF3cL_mbfq-U1M2WnI5kDPYEt8Jy5kirsg'

    telegram_user_id = 30723327  # replace None with your telegram user id (integer):

    # Create a DLBot instance
    bot = DLBot(token=telegram_token, user_id=telegram_user_id, net_name=net_name)
    # Create a TelegramBotCallback instance
    telegram_callback = TelegramBotCallback(bot)

    return telegram_callback


def superconvergence_training(model, train_gn, val_gn, test_gn, net_name,
                              batch_size=128,
                              max_lr=0.01,
                              epochs=30,
                              patience=4,
                              model_checkpoint=1):
    model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])
    model.summary()

    nowstr = time.strftime('%Y-%m-%d_%H-%M-%S')
    net_name_time = f"{net_name}_{nowstr}"

    check_path = f'output_data/checkpoints/{net_name_time}.hdf5'
    check = ModelCheckpoint(filepath=check_path, save_best_only=True, period=model_checkpoint)
    clr = OneCycleLR(max_lr=max_lr,
                     num_epochs=epochs,
                     num_samples=len(train_gn),
                     batch_size=batch_size)

    stop = EarlyStopping(patience=patience)
    # tg = get_telegram_callback(net_name)

    result = model.fit_generator(generator=train_gn,
                                 validation_data=val_gn,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[check, clr, stop],  # tg],
                                 use_multiprocessing=False,
                                 workers=1)

    result_path = f'output_data/loss_history/{net_name_time}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    # y_pred_test = model.predict_generator(generator=test_gn,
    #                                       verbose=1,
    #                                       use_multiprocessing=False,
    #                                       workers=1)

    # reconstructions_path = f'output_data/reconstructions/{net_name_time}.pkl'
    # with open(reconstructions_path, 'wb') as f:
    #     pickle.dump(y_pred_test, f)

    return result  # , y_pred_test


def snapshot_training(model, train_gn, val_gn, net_name, max_lr=0.01, epochs=10, snapshot_number=5, task='direction',
                      do_telegram=True, machine='towerino', test_gn=None):
    if task == 'direction':
        model.compile(optimizer='sgd', loss='mse')
    elif task == 'energy':
        model.compile(optimizer='sgd', loss='mse')
    elif task == 'separation':
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])

    nowstr = time.strftime('%Y-%m-%d_%H-%M-%S')
    net_name_time = f"{net_name}_{nowstr}"

    snapshot = SnapshotCallbackBuilder(epochs, snapshot_number, max_lr)
    callbacks = snapshot.get_callbacks(model_prefix=net_name_time)
    if do_telegram:
        tg = get_telegram_callback(net_name, machine=machine)
        callbacks.append(tg)
    logger = CSVLogger(f'output_data/csv_logs/{net_name_time}.csv')
    callbacks.append(logger)
    if machine == 'towerino':
        result = model.fit_generator(generator=train_gn,
                                     validation_data=val_gn,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     use_multiprocessing=False,
                                     workers=1)
        print('Training completed')
    elif machine == '24cores':
        result = model.fit_generator(generator=train_gn,
                                     validation_data=val_gn,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     use_multiprocessing=True,
                                     workers=24)
    result_path = f'output_data/loss_history/{net_name_time}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print('Result saved')
    if test_gn is not None:
        y_pred_test = model.predict_generator(generator=test_gn,
                                              verbose=1,
                                              use_multiprocessing=False,
                                              workers=8)

        print('Saving predictions...')
        reconstructions_path = f'output_data/reconstructions/{net_name_time}.pkl'
        with open(reconstructions_path, 'wb') as f:
            pickle.dump(y_pred_test, f)
        print('saved')

    return result, y_pred_test
