import os
import numpy as np
import xgboost as xgb
import glob
import pandas as pd

def load_station_fill_nan_model(station_name):
    model = xgb.XGBRegressor()
    model.load_model(f"./saved/fill_nan_models/{station_name}.json")
    return model


if __name__ == "__main__":
    all_csv = glob.glob("./data/raw/data-train/input/*.csv") + glob.glob("./data/raw/data-train/output/*.csv")
    for csv_file in all_csv:
        station_name = os.path.basename(csv_file).split(".")[0]
        model = load_station_fill_nan_model(station_name)

        df = pd.read_csv(csv_file)
        df_pm25 = pd.DataFrame(df["PM2.5"])
        df_remain = pd.DataFrame(df[["humidity", "temperature"]])
        df_timestamp = pd.DataFrame(df["timestamp"])

        df_tmp1 = df_remain.rolling(2).mean()
        df_tmp2 = df_remain.iloc[::-1].rolling(2).mean()
        df_tmp3 = df_tmp1.fillna(df_tmp2).fillna(df_remain).interpolate(method='nearest').ffill().bfill()
        df_remain = df_remain.fillna(df_tmp3)
        
        for index, row in df_pm25.iterrows():
            if (pd.isna(row["PM2.5"])):
                row["PM2.5"] = model.predict(np.array([df_remain.iloc[index].values]))
        
        final_df = pd.concat([df_timestamp, df_pm25, df_remain], axis=1)

        new_csv_file = csv_file.replace("raw", "processed_new_fill_nan")
        print(new_csv_file)
        final_df.to_csv(new_csv_file)

