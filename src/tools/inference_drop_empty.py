import numpy as np
import pandas as pd
import glob
import argparse
import yaml
import os
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from air_forecaster.utils.getter import get_instance


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/config.yaml")
    parser.add_argument("--weights_folder", default="get_latest_folder")
    return parser.parse_args()

def load_config(config_path):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    return config

def load_input_data(data_dir):
    df = pd.read_csv(data_dir)
    data = df['PM2.5'].values.reshape(1, -1)
    return data

def check_all_null(data_dir):
    df = pd.read_csv(data_dir)
    null_25 = df['PM2.5'].isnull().sum()
    return (null_25 == len(df['PM2.5']))

def load_models(weights_folder):
    all_models = {}
    config = load_config(os.path.join(weights_folder, "train_configs.yaml"))
    for model_name in config['models']:
        model_config = config['models'][model_name]
        station_models = {}

        for station_weight_path in glob.glob(os.path.join(weights_folder, "models", model_name, "*")):
            # load specific model
            model = get_instance(model_config)
            model.load_model(station_weight_path)
            station_name = os.path.basename(station_weight_path).split(".")[0]
            station_models[station_name] = model

        all_models[model_name] = {
            "result_ratio": model_config["result_ratio"],
            "station_models": station_models
        }
    return all_models

def generate_forecast_in_n_hour(model, history):
    # Recursively predict t steps
    _history = history
    predictions = []
    for i in range(24):
        prediction = model.predict(_history).tolist()[0]
        predictions.append(prediction)

        _temp = _history.reshape(-1).tolist()
        _temp = _temp[1:]
        _temp.append(prediction)

        _history = np.array(_temp).reshape(1, -1)

    return predictions

def forecast_day_based_on_neighbor(X, y, features, n_neighbor):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbor, weights = 'distance', n_jobs = -1)
    neigh.fit(X, y)
    return neigh.predict(features)

def export_submission(k_fold, list_df):
    for index, df in enumerate(list_df):
        name = 'res' + '_' + str(k_fold) + '_' + str(index + 1) + '.csv'
        path = os.path.join('results', k_fold, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame({'PM2.5': df.tolist()})
        df.to_csv(path, index=False)

def inference(args):
    if (args.weights_folder == "get_latest_folder"):
        args.weights_folder = max(glob.glob(os.path.join("saved", '*/')), key=os.path.getmtime)
    config = load_config(args.config_path)
    all_models = load_models(args.weights_folder)
    test_dir = config['dataset']['test']['args']['path']

    print('Testing...')
    for k_dir in tqdm(os.listdir(test_dir)):
        # data per fold
        data = pd.read_csv('data/train/raw/location.csv')
        data = data.assign(T1="", T2="", T3="", T4="", T5="", T6="", T7="", T8="", T9="", T10="", T11="", T12="",
                           T13="", T14="", T15="", T16="", T17="", T18="", T19="", T20="", T21="", T22="", T23="", T24="",)

        list_df = []
        for model_name in all_models:
            model_result_ratio = all_models[model_name]["result_ratio"]
            station_models = all_models[model_name]["station_models"]

            for file_name in os.listdir(os.path.join(test_dir, k_dir)):
                station_name = file_name.split(".")[0]
                full_file_name = os.path.join(test_dir, k_dir, file_name)
        
                if check_all_null(full_file_name):
                    data = data.drop(data.index[data['station'] == station_name])
                else:
                    input_data = load_input_data(full_file_name)
                    result = generate_forecast_in_n_hour(station_models[station_name], input_data)
                    data.loc[data['station'] == station_name, -24:] = result

            target_stations = data['station'].tolist()[-4:]
            for idx, target_station in enumerate(target_stations):
                lon = data.loc[data['station'] == target_station]['longitude'].tolist()[0]
                lat = data.loc[data['station'] == target_station]['latitude'].tolist()[0]
                target_feature = [[lon, lat]]
                for today in data.columns[4:]:
                    # Build data
                    X = [[x, y] for x, y in zip(data['longitude'].tolist(), data['latitude'].tolist())][:-4]
                    y = data[today].tolist()[:-4]
                    n_neighbor = config['dataset']['test']['args']['n_neighbor']
                    data.loc[data['station'] == target_station,
                            today] = forecast_day_based_on_neighbor(X, y, target_feature, n_neighbor)

            new_list_df = [data.iloc[i, -24:] for i in range(-4, 0)]
            if (list_df == []):
                list_df = new_list_df
            else:
                for i in range(len(list_df)):
                    list_df[i] += (model_result_ratio * new_list_df[i])
        export_submission(k_dir, list_df)

if __name__ == '__main__':
    args = get_parser()
    inference(args)