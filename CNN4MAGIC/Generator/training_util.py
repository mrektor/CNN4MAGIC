import time

from keras.callbacks import ModelCheckpoint, EarlyStopping

from CNN4MAGIC.CNN_Models.BigData.clr import OneCycleLR
from CNN4MAGIC.CNN_Models.BigData.snapshot import SnapshotCallbackBuilder
from CNN4MAGIC.Generator.dl_bot import DLBot
from CNN4MAGIC.Generator.telegram_bot_callback import TelegramBotCallback


def get_telegram_callback():
    # Telegram Bot imports
    telegram_token = "705094604:AAG8zNsLbcpExd_Ezhhw5TcHmgnZ---5PqM"  # replace TOKEN with your bot's token

    #  user id is optional, however highly recommended as it limits the access to you alone.
    telegram_user_id = 30723327  # replace None with your telegram user id (integer):

    # Create a DLBot instance
    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
    # Create a TelegramBotCallback instance
    telegram_callback = TelegramBotCallback(bot)

    return telegram_callback


def superconvergence_training(model, train_gn, val_gn, net_name, batch_size=128, max_lr=0.01, epochs=30, patience=5,
                              model_checkpoint=1):
    model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])

    nowstr = time.strftime('%Y-%m-%d_%H-%M-%S')
    net_name_time = f"{net_name}_{nowstr}"

    check_path = f'output_data/checkpoints/{net_name_time}.hdf5'
    check = ModelCheckpoint(filepath=check_path, save_best_only=True, period=model_checkpoint)
    clr = OneCycleLR(max_lr=max_lr,
                     num_epochs=epochs,
                     num_samples=len(train_gn),
                     batch_size=batch_size)

    stop = EarlyStopping(patience=patience)
    tg = get_telegram_callback()

    result = model.fit_generator(generator=train_gn,
                                 validation_data=val_gn,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[check, clr, stop, tg],
                                 use_multiprocessing=False,
                                 workers=3)

    return result


def snapshot_training(model, train_gn, val_gn, net_name, max_lr=0.01, epochs=10, snapshot_number=5):
    model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape'])

    nowstr = time.strftime('%Y-%m-%d_%H-%M-%S')
    net_name_time = f"{net_name}_{nowstr}"

    snapshot = SnapshotCallbackBuilder(epochs, snapshot_number, max_lr)
    callbacks = snapshot.get_callbacks(model_prefix=net_name_time)
    tg = get_telegram_callback()
    callbacks.append(tg)
    result = model.fit_generator(generator=train_gn,
                                 validation_data=val_gn,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 use_multiprocessing=False,
                                 workers=3)

    return result
