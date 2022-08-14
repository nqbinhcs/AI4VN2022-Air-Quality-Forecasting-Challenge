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
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[None]),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(1024, return_sequences=True)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(512, return_sequences=True)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(1),
        ])

        self.early_stopping = EarlyStopping()

        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(
                               learning_rate=0.001),
                           metrics=["mae"])

    def reformat_data(self, X, y, eval_set):
        X_valid, y_valid = eval_set[1]
        series_train, series_valid = [*X[0], *y], [*X_valid[0], *y_valid]
        return self.windowed_dataset(series_train), self.windowed_dataset(series_valid)

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

        _X = np.array(X[0]).reshape(1, -1)

        series_train, series_valid = self.reformat_data(X, y, eval_set)
        history = self.model.fit(
            series_train, epochs=1, validation_data=series_valid, callbacks=[self.early_stopping])

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

        window_size = 168
        _X = X[0].tolist()

        ds = tf.data.Dataset.from_tensor_slices(_X)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(32).prefetch(1)

        return self.model.predict(ds).reshape(1,)
