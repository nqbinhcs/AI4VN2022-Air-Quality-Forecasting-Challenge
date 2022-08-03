import xgboost as xgb
from abc import abstractmethod


class BaseModel():
    @abstractmethod
    def fit(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

class XGBoost(BaseModel):
    def __init__(self, n_estimators=200,
                 max_depth=7,
                 max_features='auto',
                 min_samples_split=7,
                 min_samples_leaf=3,
                 learning_rate=0.15):

        self.model = xgb.XGBRegressor(n_estimators=n_estimators,  # 200
                                      max_depth=max_depth,  # 7
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,  # 3
                                      learning_rate=learning_rate)  # 0.15

    def fit(self, X, y, eval_set, early_stopping_rounds):
        self.model.fit(X, y, eval_set=eval_set,
                       early_stopping_rounds=early_stopping_rounds)

    def save_model(self, save_dir_model):
        self.model.save_model(save_dir_model)
