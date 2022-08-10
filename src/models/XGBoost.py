from models.base import BaseModel
import xgboost as xgb


# class XGBoost(BaseModel):
#     def __init__(self, n_estimators=200,
#                  max_depth=7,
#                  max_features='auto',
#                  min_samples_split=7,
#                  min_samples_leaf=3,
#                  learning_rate=0.15):

#         self.model = xgb.XGBRegressor(n_estimators=n_estimators,  # 200
#                                       max_depth=max_depth,  # 7
#                                       max_features=max_features,
#                                       min_samples_split=min_samples_split,
#                                       min_samples_leaf=min_samples_leaf,  # 3
#                                       learning_rate=learning_rate,
#                                       gamma = 1,
#                                       reg_alpha = 0,
#                                       reg_lambda = 1
#                                       )  # 0.15

#     def fit(self, X, y, eval_set, early_stopping_rounds):
#         self.model.fit(X, y, eval_set=eval_set,
#                        early_stopping_rounds=early_stopping_rounds)

#     def save_model(self, save_dir_model):
#         self.model.save_model(save_dir_model)


class XGBoost(BaseModel):
    def __init__(self, parameters):
        self.model = xgb.XGBRegressor(**parameters)  # 0.15

    def fit(self, X, y, eval_set, early_stopping_rounds):
        self.model.fit(X, y, eval_set=eval_set,
                       early_stopping_rounds=early_stopping_rounds)

    def save_model(self, save_dir_model):
        self.model.save_model(save_dir_model)

    def load_model(self, dir):
        self.model.load_model(dir)
