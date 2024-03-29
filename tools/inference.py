from ast import parse
import numpy as np
import pandas as pd
import glob
import argparse
import yaml
import os
from tqdm import tqdm

from sklearn.neighbors import KNeighborsRegressor
from pykrige.uk import UniversalKriging
from pyinterpolate.idw import inverse_distance_weighting

from air_forecaster.utils.getter import get_instance


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="public-test")
    parser.add_argument("--weights_folder", default="get_latest_folder")
    parser.add_argument("--results_folder", default="data/final-results")
    parser.add_argument(
        "--inference_method",
        default="kriging",
        help="Available methods: knn, kriging, idw",
    )
    return parser.parse_args()


def load_config(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    return config


def load_input_data(data_dir, use_lstm):
    if use_lstm:
        df = pd.read_csv(data_dir)
        pm = df["PM2.5"].values
        humidity = df["humidity"].values
        temperature = df["temperature"].values
        # reshape
        pm = pm.reshape((pm.shape[0], 1))
        humidity = humidity.reshape((humidity.shape[0], 1))
        temperature = temperature.reshape((temperature.shape[0], 1))
        dataset = np.hstack((humidity, temperature, pm))
        history = dataset.reshape((-1, 168, 3))
        return history
    else:
        df = pd.read_csv(data_dir)
        data = df["PM2.5"].values.reshape(1, -1)
        return data


def load_models(weights_folder):
    all_models = {}
    config = load_config(os.path.join(weights_folder, "train_config.yaml"))
    for model_name in config["models"]:
        model_config = config["models"][model_name]
        station_models = {}
        for station_weight_path in glob.glob(
            os.path.join(weights_folder, "models", model_name, "*")
        ):
            # load specific model
            model = get_instance(model_config)
            model.load_model(station_weight_path)
            station_name = os.path.basename(station_weight_path).split(".")[0]
            station_models[station_name] = model
        all_models[model_name] = {
            "result_ratio": model_config["result_ratio"],
            "station_models": station_models,
        }
    return all_models


def check_all_null(data_dir):
    df = pd.read_csv(data_dir)
    null_pm25 = df["PM2.5"].isnull().sum()
    null_temperature = df["temperature"].isnull().sum()
    null_humidity = df["humidity"].isnull().sum()
    return (
        null_pm25 == len(df["PM2.5"])
        or null_temperature == len(df["temperature"])
        or null_humidity == len(df["humidity"])
    )


def generate_forecast_in_n_hour(model, history, use_lstm, hour=24):
    # Recursively predict t steps
    if use_lstm:
        y_hat = [x[2] for x in model.predict(history).tolist()[0]]
        return y_hat
    else:
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


def forecast_day_based_on_knn(X, y, target_station_coordinates, n_neighbors):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    neigh.fit(X, y)
    return neigh.predict(target_station_coordinates)


def forecast_day_based_on_kriging(X, y, target_station_coordinates):
    data = np.array([[X[idx][0], X[idx][1], y[idx]] for idx in range(len(X))])
    OK = UniversalKriging(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        variogram_model="exponential",
        verbose=False,
        enable_plotting=False,
        nlags=1,
    )
    z, ss = OK.execute(
        "points", target_station_coordinates[0][0], target_station_coordinates[0][1]
    )
    return [z[0] + 7.5]


def forecast_day_based_on_idw(X, y, target_features):
    data = np.array([[X[idx][0], X[idx][1], y[idx]] for idx in range(len(X))])
    idw_results = inverse_distance_weighting(
        data, target_features, number_of_neighbours=-1, power=2.0
    )
    return idw_results + 1.0


def export_submission(k_fold, list_df, results_folder):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for index, df in enumerate(list_df):
        name = "res" + "_" + str(k_fold) + "_" + str(index + 1) + ".csv"
        path = os.path.join(results_folder, k_fold, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame({"PM2.5": df.tolist()})
        df.to_csv(path, index=False)


def inference(args):
    if args.weights_folder == "get_latest_folder":
        args.weights_folder = max(
            glob.glob(os.path.join("saved", "*/")), key=os.path.getmtime
        )
    config = load_config(os.path.join(args.weights_folder, "train_config.yaml"))

    all_models = load_models(args.weights_folder)
    test_dir = os.path.join(config["dataset"]["test"]["args"]["path"], "input/")

    print("Testing...")
    for k_dir in tqdm(os.listdir(test_dir)):
        # load train location coordinates
        if args.phase == "public-test":
            data = pd.read_csv(os.path.join(args.weights_folder, "train-location.csv"))
        else:
            location_input = os.path.join(test_dir, k_dir, "location_input.csv")
            location_output = os.path.join(test_dir, k_dir, "location_output.csv")
            data_location_input = pd.read_csv(location_input)
            data_location_output = pd.read_csv(location_output)
            num_target_stations = data_location_output.shape[0]
            data = pd.concat([data_location_input, data_location_output])

        data.reset_index(inplace=True)
        data = data.assign(
            T1="",
            T2="",
            T3="",
            T4="",
            T5="",
            T6="",
            T7="",
            T8="",
            T9="",
            T10="",
            T11="",
            T12="",
            T13="",
            T14="",
            T15="",
            T16="",
            T17="",
            T18="",
            T19="",
            T20="",
            T21="",
            T22="",
            T23="",
            T24="",
        )

        list_df = []
        for model_name in all_models:
            use_lstm = model_name == "LSTM"
            model_result_ratio = all_models[model_name]["result_ratio"]
            station_models = all_models[model_name]["station_models"]

            for filepath in glob.glob(os.path.join(test_dir, k_dir, "*.csv")):
                filename = os.path.basename(filepath)
                if filename in ["location_input.csv", "location_output.csv"]:
                    continue

                station_name = filename.replace(".csv", "")

                if check_all_null(filepath) or (station_name not in station_models):
                    data = data.drop(data.index[data["location"] == station_name])
                else:
                    input_data = load_input_data(filepath, use_lstm)
                    result = generate_forecast_in_n_hour(
                        station_models[station_name], input_data, use_lstm
                    )
                    data.loc[data["location"] == station_name, -24:] = result

            target_stations = data["location"].tolist()[-num_target_stations:]
            for idx, target_station in enumerate(target_stations):
                lon = data.loc[data["location"] == target_station][
                    "longitude"
                ].tolist()[0]
                lat = data.loc[data["location"] == target_station]["latitude"].tolist()[
                    0
                ]
                target_station_coordinates = [[float(lon), float(lat)]]

                for today in data.columns[3:]:
                    # Build data
                    X = [
                        [float(x), float(y)]
                        for x, y in zip(
                            data["longitude"].tolist(), data["latitude"].tolist()
                        )
                    ][:-num_target_stations]
                    y = data[today].tolist()[:-num_target_stations]

                    if args.inference_method == "knn":
                        n_neighbors = config["dataset"]["test"]["args"]["n_neighbor"]
                        prediction = forecast_day_based_on_knn(
                            X, y, target_station_coordinates, n_neighbors
                        )
                    elif args.inference_method == "kriging":
                        prediction = forecast_day_based_on_kriging(
                            X, y, target_station_coordinates
                        )
                    elif args.inference_method == "idw":
                        prediction = forecast_day_based_on_idw(
                            X, y, target_station_coordinates
                        )
                    else:
                        raise NotImplementedError
                    data.loc[data["location"] == target_station, today] = prediction

            new_list_df = [data.iloc[i, -24:] for i in range(-num_target_stations, 0)]
            if list_df == []:
                list_df = new_list_df
            else:
                for i in range(len(list_df)):
                    list_df[i] += model_result_ratio * new_list_df[i]
        export_submission(k_dir, list_df, args.results_folder)


if __name__ == "__main__":
    args = get_parser()
    inference(args)
