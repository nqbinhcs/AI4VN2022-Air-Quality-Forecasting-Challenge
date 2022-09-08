import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def load_data(data_path, test_size, is_full=True):

    if is_full:
        df = pd.read_csv(data_path)
        pm = df['PM2.5'].values
        humidity = df['humidity'].values
        temperature = df['temperature'].values
        # reshape
        pm = pm.reshape((pm.shape[0], 1))
        humidity = humidity.reshape((humidity.shape[0], 1))
        temperature = temperature.reshape((temperature.shape[0], 1))
        dataset = np.hstack((humidity, temperature, pm))

        # dataset = pm

        n_steps_in, n_steps_out = 168, 24
        X_data, y_data = split_sequences(dataset, n_steps_in, n_steps_out)

        # Split with ratio
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_data, y_data, test_size=test_size, shuffle=False)

        print(X_data.shape, y_data.shape)
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        return X_train, X_valid, y_train, y_valid

    else:
        # Load X, Y
        df = pd.read_csv(data_path)
        data = df['PM2.5'].values
        X_data = []
        y_data = []
        for i in range(len(data) - 169):
            X_data.append(data[i:i+168])
            y_data.append(data[i+168])

        # Split with ratio
        X_train, X_valid, y_train, y_valid = train_test_split(
            np.array(X_data), np.array(y_data), test_size=test_size, shuffle=False)

        return X_train, X_valid, y_train, y_valid
