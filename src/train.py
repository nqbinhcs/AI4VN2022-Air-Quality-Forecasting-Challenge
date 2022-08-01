
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm
from sklearn.model_selection import train_test_split
import os
import warnings

# def custom_asymmetric_train(y_true, y_pred):
#     residual = (y_true-y_pred).astype("float")
#     grad = np.where(residual < 0, -2*10.0*residual, -2*residual)
#     hess = np.where(residual < 0, 2*10.0, 2.0)
#     return grad, hess

# def custom_asymmetric_valid(y_true, y_pred):
#     residual = (y_true-y_pred).astype("float")
#     loss = np.where(residual < 0, (residual**2)*10.0, residual**2)
#     return "custom_asymmetric_eval", np.mean(loss), False

def train(X, y, dir_save_model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, np.array(y), test_size=0.1, random_state=1)

    # best learning rate 0.1, test size = 0.2
    model = xgb.XGBRegressor(n_estimators=200,
                             max_depth=7,
                             max_features='auto',
                             min_samples_split=7,
                             min_samples_leaf=3,
                             learning_rate=0.15)

    # stopping 20                             
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=20)

    # model = lightgbm.LGBMRegressor()

    # # updating objective function to custom
    # # default is "regression"
    # # also adding metrics to check different scores
    # model.set_params(**{'objective': custom_asymmetric_train},
    #             metrics=["mse", 'mae'])

    # # fitting model
    # model.fit(
    #     X_train,
    #     y_train,
    #     eval_set=[(X_test, y_test)],
    #     eval_metric=custom_asymmetric_valid,
    #     verbose=False,
    # )


    print('Saving model...')
    print('Model is saved in', dir_save_model)
    os.makedirs(os.path.dirname(dir_save_model), exist_ok=True)
    model.save_model(dir_save_model)


def load_data(data_dir):
    df = pd.read_csv(data_dir)
    data = df['PM2.5'].values
    X_data = []
    y_data = []
    for i in range(len(data) - 169):
        X_data.append(data[i:i+168])
        y_data.append(data[i+168])

    return X_data, y_data


TRAINING_DATA_PATH = 'data/processed/data-train/input'

def main():
    for file_name in os.listdir(TRAINING_DATA_PATH):
        save_dir = os.path.join('saved/models', file_name[:-4] + '.json')
        X, y = load_data(os.path.join(TRAINING_DATA_PATH, file_name))
        train(X, y, save_dir)

if __name__ == '__main__':
    main()