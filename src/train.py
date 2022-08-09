import pprint
import argparse
import yaml
import os

# from torchan.utils.random_seed import set_determinism
# from torchan.utils.getter import get_data
from data.load_data import load_data
from trainer.trainer import Trainer
# from torchan.trainers import SupervisedTrainer


def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    is_save_config = True
    # train 11 models
    # each model of station is trained
    for file_train in os.listdir(config['dataset']['train']['args']['path']):
        save_dir_model = os.path.join(
            config['trainer']['model_dir'], file_train[:-4] + '.json')

        # 1: Load datasets
        data_path = os.path.join(
            config['dataset']['train']['args']['path'], file_train)

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
    parser.add_argument('--config')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # print(config['dataset'])
    train(config)
