import lightgbm as lgb
from air_forecaster.models.base import BaseModel

class LightGBM(BaseModel):
    def __init__(self, opt):
        self.model = lgb.LGBMRegressor(**opt)

    def fit(self, X, y, eval_set, early_stopping_rounds):
        self.model.fit(X, y, eval_set=eval_set,
                       early_stopping_rounds=early_stopping_rounds)

    def save_model(self, saved_model_path):
        # LightGBM save_model is quite different from XGBoost
        # LightGBM models should be saved as .txt files
        self.model.booster_.save_model(saved_model_path.replace(".json", ".txt"))

    def load_model(self, weight_path):
        self.model = lgb.Booster(model_file=weight_path)
    
    def predict(self, X):
        return self.model.predict(X)