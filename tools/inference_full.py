import numpy as np
import pandas as pd
import glob
import argparse
import yaml
import os
from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pyinterpolate.idw import inverse_distance_weighting
from air_forecaster.utils.getter import get_instance
from air_forecaster.utils.dataloader import split_sequences


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_folder", default="get_latest_folder")
    parser.add_argument("--results_folder", default="data/final-results")
    return parser.parse_args()


def load_config(config_path):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    return config


def check_all_null(data_dir):
    df = pd.read_csv(data_dir)
    # print(df)
    null_25 = df['PM2.5'].isnull().sum()
    null_temp = df['temperature'].isnull().sum()
    null_humi = df['humidity'].isnull().sum()
    return (null_25 == len(df['PM2.5']) or null_temp == len(df['temperature']) or null_humi == len(df['humidity']))

# coding


def load_input_data(data_dir):

    df = pd.read_csv(data_dir)
    pm = df['PM2.5'].values
    humidity = df['humidity'].values
    temperature = df['temperature'].values
    # reshape
    pm = pm.reshape((pm.shape[0], 1))
    humidity = humidity.reshape((humidity.shape[0], 1))
    temperature = temperature.reshape((temperature.shape[0], 1))
    dataset = np.hstack((humidity, temperature, pm))
    # dataset = pm

    history = dataset.reshape((-1, 168, 3))

    return history


def load_models(weights_folder):
    all_models = {}
    config = load_config(os.path.join(weights_folder, "train_config.yaml"))
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


def generate_forecast_in_n_hour(model, history, hour=24):
    # Recursive predict t steps
    # print('Check', history.shape)
    # print('Is Nan', np.isnan(history).all(),
    #       'Is Finite', np.isfinite(history).all())
    # print("SHAPE", history.shape)
    # print()
    y_hat = [x[2] for x in model.predict(history).tolist()[0]]
    # print("LEN", y_hat)
    return y_hat


def forecast_day_based_on_neighbor(X, y, features, n_neighbor):

    neigh = KNeighborsRegressor(
        n_neighbors=n_neighbor, weights='distance', n_jobs=-1)
    neigh.fit(X, y)
    return neigh.predict(features)


def forecast_day_based_on_idw(X, y, target_features):
    data = np.array([[X[idx][0], X[idx][1], y[idx]] for idx in range(len(X))])
    print(data)
    idw_results = inverse_distance_weighting(
        data,
        target_features,
        number_of_neighbours=-1,
        power=2.
    )
    return idw_results + 4


def forecast_day_based_on_kriging(X, y, features):
    data = np.array([[X[idx][0], X[idx][1], y[idx]] for idx in range(len(X))])
    OK = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    z, ss = OK.execute("points", features[0][0], features[0][1])
    return [z[0]]


def export_submission(k_fold, list_df, results_folder):
    for index, df in enumerate(list_df):
        name = 'res' + '_' + str(k_fold) + '_' + str(index + 1) + '.csv'
        path = os.path.join(results_folder, k_fold, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # print(df.tolist())
        df = pd.DataFrame({'PM2.5': df.tolist()})
        df.to_csv(path, index=False)


def inference(args):
    if (args.weights_folder == "get_latest_folder"):
        args.weights_folder = max(
            glob.glob(os.path.join("saved", '*/')), key=os.path.getmtime)
    config = load_config(os.path.join(
        args.weights_folder, "train_config.yaml"))
    all_models = load_models(args.weights_folder)
    test_dir = os.path.join(
        config['dataset']['test']['args']['path'], "input/")

    print('Testing...')
    for k_dir in tqdm(os.listdir(test_dir)):
        # data per fold
        location_input = os.path.join(test_dir, k_dir, 'location_input.csv')
        location_output = os.path.join(test_dir, k_dir, 'location_output.csv')

        data_location_input = pd.read_csv(location_input)
        data_location_output = pd.read_csv(location_output)

        # load train location coordinates
        # data = pd.read_csv(os.path.join(
        #     args.weights_folder, "train-location.csv"))
        data = pd.concat([data_location_input, data_location_output])

        print(data.to_string())

        data = data.assign(T1="", T2="", T3="", T4="", T5="", T6="", T7="", T8="", T9="", T10="", T11="", T12="",
                           T13="", T14="", T15="", T16="", T17="", T18="", T19="", T20="", T21="", T22="", T23="", T24="",)

        print(data['location'].tolist())
        list_df = []
        for model_name in all_models:
            print(model_name)
            model_result_ratio = all_models[model_name]["result_ratio"]
            station_models = all_models[model_name]["station_models"]

            for file_name in os.listdir(os.path.join(test_dir, k_dir)):

                # skip location files
                if file_name in ['location_input.csv', 'location_output.csv', '1', '2', '3', '4', '5', '6']:
                    continue


                station_name = file_name.replace('.csv', '')
                full_file_name = os.path.join(test_dir, k_dir, file_name)

                if (check_all_null(full_file_name)) or (station_name not in station_models):
                    data = data.drop(
                        data.index[data['location'] == station_name])
                    print('before', data['location'].tolist())
                    print('drop', station_name)
                    print('after', data['location'].tolist())
                    print('====>')

                else:
                    input_data = load_input_data(full_file_name)
                    result = generate_forecast_in_n_hour(
                        station_models[station_name], input_data)
                    data.loc[data['location'] == station_name, -24:] = result

            target_stations = data['location'].tolist()[-6:]
            print(data['location'].tolist())
            print(target_stations)

            for idx, target_station in enumerate(target_stations):
                lon = data.loc[data['location'] == target_station]['longitude'].tolist()[
                    0]
                lat = data.loc[data['location'] == target_station]['latitude'].tolist()[
                    0]
                target_feature = [[lon, lat]]

                print(data.to_string())

                for today in data.columns[3:]:
                    # Build data
                    X = [[x, y] for x, y in zip(
                        data['longitude'].tolist(), data['latitude'].tolist())][:-6]
                    y = data[today].tolist()[:-6]
                    n_neighbor = config['dataset']['test']['args']['n_neighbor']
                    # data.loc[data['location'] == target_station,
                    #          today] = forecast_day_based_on_neighbor(X, y, target_feature, n_neighbor)
                    print("k dir", k_dir, len(data))
                    data.loc[data['location'] == target_station,
                             today] = forecast_day_based_on_idw(X, y, target_feature)

            new_list_df = [data.iloc[i, -24:] for i in range(-6, 0)]
            if (list_df == []):
                list_df = new_list_df
            else:
                for i in range(len(list_df)):
                    list_df[i] += (model_result_ratio * new_list_df[i])

        export_submission(k_dir, list_df, args.results_folder)


if __name__ == '__main__':
    args = get_parser()
    inference(args)
