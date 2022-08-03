import pprint
import argparse
import yaml
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from trainer import Trainer

def load_data(data_path, test_size):
    df = pd.read_csv(data_path)
    df = df.iloc[: , 1:]
    df = df[~df.isnull().any(axis=1)] # select full infos rows
    X_data = np.array([_ for _ in zip(df["humidity"].values, df["temperature"].values)])
    y_data = np.array(df["PM2.5"].values)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_data, y_data, test_size=test_size, shuffle=False, random_state=7)
    return X_train, X_valid, y_train, y_valid

def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    is_save_config = True
    # train 15 models
    # each model of station is trained
    for path in config['dataset']['train']['args']['paths']:
        for file_train in os.listdir(path):
            save_dir_model = os.path.join(
                config['trainer']['model_dir'], file_train[:-4] + '.json')

            # 1: Load datasets
            data_path = os.path.join(path, file_train)

            X_train, X_valid, y_train, y_valid = load_data(
                data_path=data_path, test_size=config['dataset']['train']['args']['ratio'])

            # 2: Create trainer
            trainer = Trainer(config)

            # 3: Start trainining
            trainer.train(
                (X_train, y_train), (X_valid, y_valid), save_dir_model, is_save_config)

            # No needed to save config file of model, because 11 models have the same config file
            is_save_config = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')

    args = parser.parse_args()

    config_path = args.config_file
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # print(config['dataset'])
    train(config)

    