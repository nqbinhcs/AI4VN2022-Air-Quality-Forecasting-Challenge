import pprint
import argparse
import yaml
import os
import glob

from air_forecaster.utils.dataloader import load_data
from air_forecaster.trainer import Trainer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/config.yaml")
    parser.add_argument("--saved_runs_folder", default="saved/")
    return parser.parse_args()


def train(args):
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    pprint.PrettyPrinter(indent=2).pprint(config)

    train_id = f"{config['method_name']}_{len(glob.glob(os.path.join(args.saved_runs_folder, '*/')))}"
    if not os.path.exists(os.path.join(args.saved_runs_folder, train_id)):
        os.makedirs(os.path.join(args.saved_runs_folder, train_id))

    # save train config for model loading when inference
    save_config = True
    if (save_config):
        saved_config_path = os.path.join(
            args.saved_runs_folder, train_id, "train_config.yaml")
        with open(saved_config_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    for station_data_path in glob.glob(os.path.join(config['dataset']['train']['args']['path'], "input/*")):
        station_name = os.path.basename(station_data_path).split(".")[0]

        # Step 1: Load datasets
        X_train, X_valid, y_train, y_valid = load_data(
            data_path=station_data_path, test_size=config['dataset']['train']['args']['split_ratio'])

        # Step 2: create trainers
        for model_name in config['models']:
            model_config = config['models'][model_name]
            saved_model_path = os.path.join(
                args.saved_runs_folder, train_id, "models", model_name, station_name + ".json")

            # create each trainer for each type of model
            print("Create", model_name, " trainer for ", station_name)
            trainer = Trainer(model_config)

            # start training
            print("X_train shape", X_train.shape)
            print("y_train shape", y_train.shape)
            trainer.train((X_train, y_train),
                          (X_valid, y_valid), saved_model_path)


if __name__ == "__main__":
    args = get_parser()
    train(args)
