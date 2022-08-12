from utils.getter import get_instance
from tqdm import tqdm
import yaml
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(".")


class Trainer:
    def __init__(self, config):
        super().__init__()

        self.load_config_dict(config)
        self.config = config

        # Train ID
        self.train_id = self.config.get('id', 'None')
        self.train_id += '-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    def load_config_dict(self, config):
        self.model = get_instance(config['model'])

    def save_config(self, path):
        with open(path, 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def train(self, train_dataloader, val_dataloader, save_dir_model, is_save_config):

        X_train, y_train = train_dataloader
        X_valid, y_valid = val_dataloader

        # 20
        self.model.fit(X_train, y_train,
                       eval_set=[(X_train, y_train), (X_valid, y_valid)],
                       early_stopping_rounds=20)

        # self.model.fit(X_train, y_train,
        #                eval_set=(X_valid, y_valid),
        #                early_stopping_rounds=20)

        # Save model

        print('Saving model...')
        print('Model is saved in', save_dir_model)
        os.makedirs(os.path.dirname(save_dir_model), exist_ok=True)
        self.model.save_model(save_dir_model)

        # Save config
        if is_save_config:
            path_config = os.path.join(
                'saved', 'logs', self.train_id, 'config.yaml')
            os.makedirs(os.path.dirname(path_config), exist_ok=True)
            self.save_config(path_config)
