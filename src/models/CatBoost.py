from models.base import BaseModel
import catboost as cb


class CatBoost(BaseModel):
    def __init__(self, parameters):
        self.model = cb.CatBoostRegressor(**parameters)

    def fit(self, X, y, eval_set, early_stopping_rounds):
        self.model.fit(X, y,
                       eval_set=eval_set,
                       use_best_model=True,
                       verbose=50
                       )

    def save_model(self, save_dir_model):
        self.model.save_model(save_dir_model, format="json")

    def load_model(self, dir):
        self.model.load_model(dir, "json")

    def predict(self, X):
        return self.model.predict(X)
