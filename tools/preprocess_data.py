import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob
from shutil import copy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        default="public-test", help="Available types: public-test, private-test")
    parser.add_argument("--method",
                        default="mean", help="Available methods: mean, interpolate")
    parser.add_argument("--data_train_folder_path", default="data/train/raw")
    parser.add_argument("--preprocessed_data_train_folder_path",
                        default="data/train/preprocessed")
    parser.add_argument("--public_test_folder_path",
                        default="data/public-test/raw")
    parser.add_argument("--preprocessed_public_test_folder_path",
                        default="data/public-test/preprocessed")

    return parser.parse_args()


def convert_to_true_format(input_file_path, type):
    if type == 'private-test':
        df = pd.read_csv(input_file_path)
        df.rename(columns={'time': 'timestamp'}, inplace=True)
        df.to_csv(input_file_path, index=False)


def create_date_features(df):
    df['month'] = df.timestamp.dt.month
    df['day_of_month'] = df.timestamp.dt.day
    df['day_of_year'] = df.timestamp.dt.dayofyear
    df['week_of_year'] = df.timestamp.dt.weekofyear
    # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
    df['day_of_week'] = df.timestamp.dt.dayofweek + 1
    df['year'] = df.timestamp.dt.year
    df["is_wknd"] = df.timestamp.dt.weekday // 4
    df['is_month_start'] = df.timestamp.dt.is_month_start.astype(int)
    df['is_month_end'] = df.timestamp.dt.is_month_end.astype(int)
    return df


def process_data_mean(input_file_path, output_file_path):

    df = pd.read_csv(input_file_path)
    df = df.iloc[:, 1:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['PM2.5'].fillna(value=df['PM2.5'].mean(), inplace=True)
    df['humidity'].fillna(value=df['humidity'].mean(), inplace=True)
    df['temperature'].fillna(value=df['temperature'].mean(), inplace=True)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)


def process_data_interpolate(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    df = df.iloc[:, 1:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['PM2.5'].interpolate(method='linear', inplace=True)
    df['humidity'].interpolate(method='linear', inplace=True)
    df['temperature'].interpolate(method='linear', inplace=True)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)


def preprocess_data(args):
    type = args.type
    method = args.method
    data_train_folder_path = args.data_train_folder_path
    preprocessed_data_train_folder_path = args.preprocessed_data_train_folder_path
    public_test_folder_path = args.public_test_folder_path
    preprocessed_public_test_folder_path = args.preprocessed_public_test_folder_path

    print(f'Type data: {type}')
    print(f'Method: {method}')

    # preprocessing data train
    # need_to_preprocess_sub_dirs = [
    #     "input", "output"] if type == 'public-test' else ["input"]
    # for sub_dir in need_to_preprocess_sub_dirs:
    #     print(f"Preprocessing files in {sub_dir} in data train...")
    #     for input_file_path in tqdm(glob.glob(os.path.join(data_train_folder_path, f"{sub_dir}/*"))):
    #         file_name = os.path.basename(input_file_path)
    #         output_file_path = os.path.join(
    #             preprocessed_data_train_folder_path, f"{sub_dir}", file_name)

    #         # convert to true format if private test
    #         convert_to_true_format(input_file_path, type)

    #         if method == 'interpolate':
    #             process_data_interpolate(input_file_path, output_file_path)
    #         elif method == 'mean':
    #             process_data_mean(input_file_path, output_file_path)

    # # copy `location.csv` file to the new preprocessed data train folder
    # copy(os.path.join(data_train_folder_path, "location.csv"),
    #      preprocessed_data_train_folder_path)

    # preprocessing public test
    # `input` files
    if type == 'public-test':
        print(f"Preprocessing files in data test...")
        for sub_dir in tqdm(glob.glob(os.path.join(public_test_folder_path, "input/*/"))):
            for input_file_path in glob.glob(sub_dir + "*"):
                file_name = os.path.basename(input_file_path)
                sub_dir_name = os.path.basename(os.path.normpath(sub_dir))
                output_file_path = os.path.join(
                    preprocessed_public_test_folder_path, "input", sub_dir_name, file_name)
                process_data_mean(input_file_path, output_file_path)
                # process_data_interpolate(input_file_path, output_file_path)
        # copy `location.csv` file to the new preprocessed public test folder
        copy(os.path.join(public_test_folder_path, "location.csv"),
             preprocessed_public_test_folder_path)
    else:
        print(f"Preprocessing files in data test...")
        for sub_dir in tqdm(glob.glob(os.path.join(public_test_folder_path, "input/*/"))):
            for input_file_path in glob.glob(sub_dir + "*"):
                file_name = os.path.basename(input_file_path)

                # skip processing if meteo
                if file_name == 'meteo':
                    continue

                # copy location
                if file_name in ['location_input.csv', 'location_output.csv']:
                    copy(os.path.join(sub_dir, file_name),
                         os.path.join(preprocessed_public_test_folder_path, "input", sub_dir_name, file_name))
                    continue

                sub_dir_name = os.path.basename(os.path.normpath(sub_dir))
                output_file_path = os.path.join(
                    preprocessed_public_test_folder_path, "input", sub_dir_name, file_name)

                convert_to_true_format(input_file_path, type)
                process_data_mean(input_file_path, output_file_path)
                # process_data_interpolate(input_file_path, output_file_path)


if __name__ == "__main__":
    args = get_parser()
    preprocess_data(args)
