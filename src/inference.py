import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pprint
import argparse
import yaml
import os
from tqdm import tqdm
import warnings
from utils.getter import get_instance


def load_input_data(data_dir):
    df = pd.read_csv(data_dir)
    data = df['PM2.5'].values.reshape(1, -1)
    return data


def check_null_all(data_dir):
    df = pd.read_csv(data_dir)
    null_25 = df['PM2.5'].isnull().sum()
    return (null_25 == len(df['PM2.5']))


def load_models(config):
    models = dict()
    saved_dir = 'saved/models'
    for model_name in os.listdir(saved_dir):
        model = get_instance(config['model'])
        model.load_model(os.path.join(saved_dir, model_name))

        # edit
        model_name = model_name.split('.')[0]
        models[model_name] = model
        print(f'Loaded model {model_name}')
    return models


def generate_forcast_in_n_hour(model, history, hour=24):
    # Recursive predict t steps
    _history = history
    preds = []
    for i in range(24):

        prediction = model.predict(_history).tolist()[0]
        preds.append(prediction)

        _temp = _history.reshape(-1).tolist()
        _temp = _temp[1:]
        _temp.append(prediction)

        _history = np.array(_temp).reshape(1, -1)

    return preds


def forcast_day_based_on_neighbor(X, y, features, Kneighbor):
    neigh = KNeighborsRegressor(
        n_neighbors=Kneighbor, weights='distance', n_jobs=-1)

    # print(X)
    # print(y)
    # print(features)

    neigh.fit(X, y)
    return neigh.predict(features)


def to_submission(k_fold, list_df):
    for idx, df in enumerate(list_df):
        name = 'res' + '_' + str(k_fold) + '_' + str(idx+1) + '.csv'
        path = os.path.join('results', k_fold, name)
        # path = os.path.join('results', 'CatBoost', k_fold, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # print(df.tolist())
        df = pd.DataFrame({'PM2.5': df.tolist()})
        df.to_csv(path, index=False)


def inference(config):

    models = load_models(config)

    test_dir = config['dataset']['inference']['args']['path']

    print('Inferencing...')
    for k_dir in tqdm(os.listdir(test_dir)):
        # data per fold
        data = pd.read_csv('data/raw/data-train/location.csv')

        data = data.assign(T1="", T2="", T3="", T4="", T5="", T6="", T7="", T8="", T9="", T10="", T11="", T12="",
                           T13="", T14="", T15="", T16="", T17="", T18="", T19="", T20="", T21="", T22="", T23="", T24="",)

        for file_name in os.listdir(os.path.join(test_dir, k_dir)):
            model_name = file_name[:-4]
            full_file_name = os.path.join(test_dir, k_dir, file_name)

            if check_null_all(full_file_name):
                # print('Drop', file_name)
                # print(file_name.split('.')[0])
                data = data.drop(
                    data.index[data['station'] == file_name.split('.')[0]])
            else:

                input_data = load_input_data(full_file_name)
                # print(input_data.shape)

                result = generate_forcast_in_n_hour(models[model_name], input_data)

                data.loc[data['station'] == model_name, -24:] = result

        # print(data.to_string())

        target_stations = data['station'].tolist()[-4:]
        for idx, target_station in enumerate(target_stations):
            lon = data.loc[data['station'] == target_station]['longitude'].tolist()[
                0]
            lat = data.loc[data['station'] == target_station]['latitude'].tolist()[
                0]
            target_feature = [[lon, lat]]
            for Tday in data.columns[4:]:
                # Build data
                X = [[x, y] for x, y in zip(
                    data['longitude'].tolist(), data['latitude'].tolist())][:-4]
                y = data[Tday].tolist()[:-4]
                data.loc[data['station'] == target_station,
                         Tday] = forcast_day_based_on_neighbor(X, y, target_feature, config['dataset']['inference']['args']['Kneighbor'])
        list_df = [data.iloc[i, -24:] for i in range(-4, 0)]

        to_submission(k_dir, list_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    inference(config)
