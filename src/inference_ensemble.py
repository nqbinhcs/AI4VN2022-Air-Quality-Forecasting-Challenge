import pandas as pd
import os
import random


# 0.3 0.7
# 0.5 0.5
# 0.8 0.2

random.seed(1112)


def ensemble(weights=[0.01, 0.01, 0.98]):
    for k_folder in range(1, 101):
        for k_file in range(1, 5):

            lstm_file = 'results/LSTM/{}/res_{}_{}.csv'.format(
                k_folder, k_folder, k_file)
            xgb_file = 'results/XGBoost/{}/res_{}_{}.csv'.format(
                k_folder, k_folder, k_file)
            cb_file = 'results/CatBoost/{}/res_{}_{}.csv'.format(
                k_folder, k_folder, k_file)

            ensemble_file = 'results/Ensemble/{}/res_{}_{}.csv'.format(
                k_folder, k_folder, k_file)

            df_xgb = pd.read_csv(xgb_file)
            df_cb = pd.read_csv(cb_file)
            df_lstm = pd.read_csv(lstm_file)

            df_ss = pd.DataFrame(
                {"XGB": df_xgb["PM2.5"], "CB": df_cb["PM2.5"], "LSTM": df_lstm["PM2.5"]})

            print(ensemble_file, random.randrange(0, 2))
            print(df_ss.to_string())

            # if random.randrange(0, 2) == 0:
            #     df_ensemble = df_xgb["PM2.5"]
            # else:
            #     df_ensemble = df_lstm["PM2.5"]

            # df_ensemble = df_xgb["PM2.5"] * \
            #     weights[0] + df_lstm["PM2.5"] * weights[1]

            df_ensemble = df_xgb["PM2.5"] * \
                weights[0] + df_cb["PM2.5"] * weights[1] + \
                df_lstm["PM2.5"] * weights[2]

            # df_ensemble = (df_xgb["PM2.5"] +
            #                df_cb["PM2.5"] + df_lstm["PM2.5"])/3

            df = pd.DataFrame({'PM2.5': df_ensemble.tolist()})
            os.makedirs(os.path.dirname(ensemble_file), exist_ok=True)
            df.to_csv(ensemble_file, index=False)


ensemble()
