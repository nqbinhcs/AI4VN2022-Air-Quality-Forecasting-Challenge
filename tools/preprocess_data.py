import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob
from shutil import copy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default="public-test",
        help="Available phases: public-test, private-test",
    )
    parser.add_argument(
        "--method", default="mean", help="Available methods: mean, interpolate"
    )
    parser.add_argument("--train_folder_path", default="data/public-train/raw")
    parser.add_argument(
        "--preprocessed_train_folder_path", default="data/public-train/preprocessed"
    )
    parser.add_argument("--test_folder_path", default="data/public-test/raw")
    parser.add_argument(
        "--preprocessed_test_folder_path", default="data/public-test/preprocessed"
    )
    return parser.parse_args()


def create_dirs(args):
    if not os.path.exists(args.train_folder_path):
        os.makedirs(args.train_folder_path)
    if not os.path.exists(args.preprocessed_train_folder_path):
        os.makedirs(args.preprocessed_train_folder_path)
    if not os.path.exists(args.test_folder_path):
        os.makedirs(args.test_folder_path)
    if not os.path.exists(args.preprocessed_test_folder_path):
        os.makedirs(args.preprocessed_test_folder_path)


def convert_to_true_format(input_file_path, phase):
    if phase == "private-test":
        df = pd.read_csv(input_file_path)
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df.to_csv(input_file_path, index=False)


def create_date_features(df):
    df["month"] = df.timestamp.dt.month
    df["day_of_month"] = df.timestamp.dt.day
    df["day_of_year"] = df.timestamp.dt.dayofyear
    df["week_of_year"] = df.timestamp.dt.weekofyear
    # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
    df["day_of_week"] = df.timestamp.dt.dayofweek + 1
    df["year"] = df.timestamp.dt.year
    df["is_wknd"] = df.timestamp.dt.weekday // 4
    df["is_month_start"] = df.timestamp.dt.is_month_start.astype(int)
    df["is_month_end"] = df.timestamp.dt.is_month_end.astype(int)
    return df


def fill_data(input_file_path, output_file_path, method):
    if method == "mean":
        df = pd.read_csv(input_file_path)
        df = df.iloc[:, 1:]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["PM2.5"].fillna(value=df["PM2.5"].mean(), inplace=True)
        df["humidity"].fillna(value=df["humidity"].mean(), inplace=True)
        df["temperature"].fillna(value=df["temperature"].mean(), inplace=True)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False)
    elif method == "interpolate":
        df = pd.read_csv(input_file_path)
        df = df.iloc[:, 1:]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["PM2.5"].interpolate(method="linear", inplace=True)
        df["humidity"].interpolate(method="linear", inplace=True)
        df["temperature"].interpolate(method="linear", inplace=True)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False)
    else:
        raise NotImplementedError


def preprocess_data(args):
    print("Preprocessing data, input args: ", args)
    create_dirs(args)

    # preprocessing data train
    need_to_preprocess_sub_dirs = (
        ["input", "output"] if args.phase == "public-test" else ["input"]
    )
    for sub_dir in need_to_preprocess_sub_dirs:
        print(f"Preprocessing files in {sub_dir} in data train...")
        for input_file_path in tqdm(
            glob.glob(os.path.join(args.train_folder_path, f"{sub_dir}/*"))
        ):
            file_name = os.path.basename(input_file_path)
            output_file_path = os.path.join(
                args.preprocessed_train_folder_path, f"{sub_dir}", file_name
            )
            # convert to true format if private test
            convert_to_true_format(input_file_path, args.phase)
            fill_data(input_file_path, output_file_path, args.method)
    # copy `location.csv` file to the new preprocessed train folder
    copy(
        os.path.join(args.train_folder_path, "location.csv"),
        args.preprocessed_train_folder_path,
    )

    # preprocessing public test
    if args.phase == "public-test":
        for sub_dir in tqdm(glob.glob(os.path.join(args.test_folder_path, "input/*/"))):
            for input_file_path in glob.glob(sub_dir + "*"):
                file_name = os.path.basename(input_file_path)
                sub_dir_name = os.path.basename(os.path.normpath(sub_dir))
                output_file_path = os.path.join(
                    args.preprocessed_test_folder_path, "input", sub_dir_name, file_name
                )
                fill_data(input_file_path, output_file_path, args.method)
        # copy `location.csv` file to the new preprocessed test folder
        copy(
            os.path.join(args.test_folder_path, "location.csv"),
            args.preprocessed_test_folder_path,
        )
    elif args.phase == "private-test":
        for sub_dir in tqdm(glob.glob(os.path.join(args.test_folder_path, "input/*/"))):
            for input_file_path in glob.glob(sub_dir + "*"):
                file_name = os.path.basename(input_file_path)
                sub_dir_name = os.path.basename(os.path.normpath(sub_dir))

                # skip processing if meteo
                if file_name == "meteo":
                    continue

                # copy location
                if file_name in ["location_input.csv", "location_output.csv"]:
                    copy(
                        os.path.join(sub_dir, file_name),
                        os.path.join(
                            args.preprocessed_test_folder_path,
                            "input",
                            sub_dir_name,
                            file_name,
                        ),
                    )
                    continue
                output_file_path = os.path.join(
                    args.preprocessed_test_folder_path, "input", sub_dir_name, file_name
                )
                convert_to_true_format(input_file_path, args.phase)
                fill_data(input_file_path, output_file_path, args.method)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = get_parser()
    preprocess_data(args)
