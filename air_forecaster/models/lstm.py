from air_forecaster.models.base import BaseModel
import numpy as np
import tensorflow as tf
import os


class EarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('mae') < 0.03):
            print("\nMAEthreshold reached. Training stopped.")
            self.model.stop_training = True


# class LSTM(BaseModel):
#     def __init__(self, parameters):
#         self.model = tf.keras.models.Sequential([
#             tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                                    input_shape=[None]),
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(1024, return_sequences=True)),
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(512, return_sequences=True)),
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(256, return_sequences=True)),
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(128, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#             tf.keras.layers.Dense(1),
#         ])

#         self.early_stopping = EarlyStopping()

#         self.model.compile(loss=tf.keras.losses.Huber(),
#                            optimizer=tf.keras.optimizers.Adam(
#                                learning_rate=0.001),
#                            metrics=["mae"])

#     def reformat_data(self, X, y, eval_set):
#         X_valid, y_valid = eval_set[1]
#         series_train, series_valid = [*X[0], *y], [*X_valid[0], *y_valid]
#         return self.windowed_dataset(series_train), self.windowed_dataset(series_valid)

#     def windowed_dataset(self, series, window_size=168, batch_size=32, shuffle_buffer=1000):
#         dataset = tf.data.Dataset.from_tensor_slices(series)
#         dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
#         dataset = dataset.flat_map(
#             lambda window: window.batch(window_size + 1))
#         dataset = dataset.shuffle(shuffle_buffer)
#         dataset = dataset.map(lambda window: (window[:-1], window[-1]))
#         dataset = dataset.batch(batch_size).prefetch(1)

#         return dataset

#     def fit(self, X, y, eval_set, early_stopping_rounds):

#         _X = np.array(X[0]).reshape(1, -1)

#         series_train, series_valid = self.reformat_data(X, y, eval_set)
#         history = self.model.fit(
#             series_train, epochs=1, validation_data=series_valid, callbacks=[self.early_stopping])

#     def save_model(self, save_dir_model):
#         dir_cp = save_dir_model.split('.')[0]
#         save_file = os.path.join(dir_cp, os.path.basename(
#             save_dir_model).split('.')[0] + '.ckpt')
#         os.makedirs(os.path.dirname(save_file), exist_ok=True)
#         self.model.save_weights(save_file)

#     def load_model(self, dir):
#         latest = tf.train.latest_checkpoint(dir)
#         self.model.load_weights(latest)

#     def predict(self, X):

#         window_size = 168
#         _X = X[0].tolist()

#         ds = tf.data.Dataset.from_tensor_slices(_X)
#         ds = ds.window(window_size, shift=1, drop_remainder=True)
#         ds = ds.flat_map(lambda w: w.batch(window_size))
#         ds = ds.batch(32).prefetch(1)

#         return self.model.predict(ds).reshape(1,)

class LSTM(BaseModel):
    def __init__(self, parameters):

        self.n_features = parameters.get('n_features')
        self.n_steps_in = parameters.get('n_steps_in')
        self.n_steps_out = parameters.get('n_steps_out')

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(256, input_shape=[
                                 parameters['n_steps_in'], parameters['n_features']]),
            tf.keras.layers.RepeatVector(parameters['n_steps_out']),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,
                                                               return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
                                                               return_sequences=True)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
            #                      return_sequences=True)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(parameters['n_features'])),
            # tf.keras.layers.Dense(parameters['n_features'])

        ])

        self.lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1)
        self.es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60,
                                                   verbose=1, mode="auto", restore_best_weights=True)

        # self.model.compile(loss=tf.keras.losses.Huber(),
        #                    optimizer=tf.keras.optimizers.Adam(),
        #                    metrics=["mae"])

        self.model.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='mae')

    def reformat_data(self, X, y, eval_set):
        X_valid, y_valid = eval_set[1]
        data_train = tf.data.Dataset.from_tensor_slices((X, y))
        data_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        data_train = data_train.shuffle(1000)
        data_train = data_train.batch(4).prefetch(1)
        data_valid = data_valid.batch(4).prefetch(1)

        def check_nan(ds):
            print('Check', ds.shape)
            print('Is Nan', np.isnan(ds).all(),
                  'Is Finite', np.isfinite(ds).all())

        # check_nan(X)
        # check_nan(y)
        # check_nan(X_valid)
        # check_nan(y_valid)

        # print('Train')
        # sequence_batch, label = next(iter(data_train))
        # print(sequence_batch)
        # print(label)

        # print('Valid')
        # sequence1_batch, label = next(iter(data_valid))
        # print(sequence1_batch)
        # print(label)

        # print('Try predict')
        # sequence_batch, label = next(iter(data_train))
        # print(self.model.predict(sequence_batch))

        return data_train, data_valid

    def windowed_dataset(self, series, window_size=168, batch_size=32, shuffle_buffer=1000):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset

    def fit(self, X, y, eval_set, early_stopping_rounds):
        data_train, data_valid = self.reformat_data(X, y, eval_set)

        # print('Try predict')
        # # sequence_batch, label = next(next(next(iter(data_train))))
        # # print(self.model.predict(sequence_batch))

        # for idx, (s, l) in enumerate(data_train):
        #     if idx == 0:
        #         print(self.model.predict(s))
        #         break

        history = self.model.fit(
            data_train, epochs=20, validation_data=data_valid, callbacks=[self.lr, self.es])

    def save_model(self, save_dir_model):
        dir_cp = save_dir_model.split('.')[0]
        save_file = os.path.join(dir_cp, os.path.basename(
            save_dir_model).split('.')[0] + '.ckpt')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        self.model.save_weights(save_file)

    def load_model(self, dir):
        latest = tf.train.latest_checkpoint(dir)
        self.model.load_weights(latest)

    def predict(self, X):

        # window_size = 168
        # _X = X[0].tolist()

        # ds = tf.data.Dataset.from_tensor_slices(_X)
        # ds = ds.window(window_size, shift=1, drop_remainder=True)
        # ds = ds.flat_map(lambda w: w.batch(window_size))
        # ds = ds.batch(32).prefetch(1)

        # return self.model.predict(ds).reshape(1,)

        X_input = X.reshape((-1, self.n_steps_in, self.n_features))
        return self.model.predict(X_input)
