import sys

sys.path.append(".")
import os
import yaml

from air_forecaster.utils.getter import get_instance


class Trainer:
    def __init__(self, model_config):
        super().__init__()

        self.load_config_dict(model_config)
        self.config = model_config

    def load_config_dict(self, model_config):
        self.model = get_instance(model_config)

    def save_config(self, saved_config_path):
        with open(saved_config_path, "w") as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def train(
        self, train_dataloader, val_dataloader, saved_model_path, saved_config_path=None
    ):
        X_train, y_train = train_dataloader
        X_valid, y_valid = val_dataloader

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=20,
        )

        # Save model
        print("Saving model...")
        print("Model is saved in", saved_model_path)
        os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
        self.model.save_model(saved_model_path)

        if saved_config_path is not None:
            os.makedirs(os.path.dirname(saved_config_path), exist_ok=True)
            self.save_config(saved_config_path)
