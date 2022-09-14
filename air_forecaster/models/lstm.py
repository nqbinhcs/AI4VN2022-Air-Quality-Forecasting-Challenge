from air_forecaster.models.base import BaseModel
import numpy as np
import tensorflow as tf
import os


class EarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('mae') < 0.03):
            print("\nMAEthreshold reached. Training stopped.")
            self.model.stop_training = True


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
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(parameters['n_features'])),

        ])

        self.lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, verbose=1)
        self.es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=60,
                                                   verbose=1, mode="auto", restore_best_weights=True)

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
        X_input = X.reshape((-1, self.n_steps_in, self.n_features))
        return self.model.predict(X_input)
