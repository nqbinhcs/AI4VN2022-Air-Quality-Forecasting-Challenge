import xgboost as xgb
from air_forecaster.models.base import BaseModel


class XGBoost(BaseModel):
    def __init__(self, opt):
        self.model = xgb.XGBRegressor(**opt)

    def fit(self, X, y, eval_set, early_stopping_rounds):
        self.model.fit(
            X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds
        )

    def save_model(self, saved_model_path):
        self.model.save_model(saved_model_path)

    def load_model(self, weight_path):
        self.model.load_model(weight_path)

    def predict(self, X):
        return self.model.predict(X)
