from models.base import BaseModel
from keras.models import Sequential
from keras.layers import LSTM as LSTML
from keras.layers import Dense
import keras
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

        # _y = self.predict(_X)
        # print(type(_y), _y)

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


# import torch
# import torch.nn as nn
# from torch.autograd import Variable

# from models.base import BaseModel


# class LSTM(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
#         super(LSTM, self).__init__()

#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.seq_length = seq_length

#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)

#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         h_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))

#         c_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))

#         # Propagate input through LSTM
#         ula, (h_out, _) = self.lstm(x, (h_0, c_0))

#         h_out = h_out.view(-1, self.hidden_size)

#         out = self.fc(h_out)

#         return out


# class LSTMmodel(BaseModel):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, num_epochs=2000, learning_rate=0.01):
#         super(LSTMmodel, self).__init__()

#         self.model = LSTM(num_classes, input_size,
#                           hidden_size, num_layers, seq_length)

#         self.num_epochs = num_epochs
#         self.learning_rate = learning_rate

#         self.criterion = torch.nn.MSELoss()
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=self.learning_rate)

#     def get_valid(self, X, y):
#         self.model.eval()
#         with torch.no_grad():
#             outputs = self.model(X)
#             loss = self.criterion(outputs, y)
#             return loss.item()

#     def fit(self, X, y, eval_set, early_stopping_rounds):

#         X_train = X
#         y_train = y
#         X_valid, y_valid = eval_set

#         for epoch in range(self.num_epochs):
#             self.model.train()

#             outputs = self.model(X_train)
#             self.optimizer.zero_grad()

#             # obtain the loss function
#             loss = self.criterion(outputs, y_train)

#             loss.backward()

#             self.optimizer.step()


#             # Early stopping
#             # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
#             # print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

#             print(
#                 f'Epoch {epoch + 1}/{self.num_epochs}, training loss = {loss.item()}, validation loss = {self.get_valid(X_valid, y_valid)}')
