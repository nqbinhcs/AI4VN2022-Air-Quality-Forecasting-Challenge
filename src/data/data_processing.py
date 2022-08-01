#!/usr/bin/env python
# coding: utf-8


from IPython.display import display
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import statsmodels as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as formula
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

sns.set()
warnings.filterwarnings(action='ignore')

INPUT_PATH = 'data/raw/public-test/input'


def create_date_features(df):
    df['month'] = df.timestamp.dt.month
    df['day_of_month'] = df.timestamp.dt.day
    df['day_of_year'] = df.timestamp.dt.dayofyear
    df['week_of_year'] = df.timestamp.dt.weekofyear
    # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
    df['day_of_week'] = df.timestamp.dt.dayofweek + 1
    df['year'] = df.timestamp.dt.year
    df["is_wknd"] = df.timestamp.dt.weekday // 4
    df['is_month_start'] = df.timestamp.dt.is_month_start.astype(int)
    df['is_month_end'] = df.timestamp.dt.is_month_end.astype(int)
    return df

def process_data(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    df = df.iloc[:, 1:]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['PM2.5'].fillna(value = df['PM2.5'].mean(), inplace = True)
    df['humidity'].fillna(value=df['humidity'].mean(), inplace=True)
    df['temperature'].fillna(value=df['temperature'].mean(), inplace=True)

    # scale if fitting to regression
    # col_names = ['humidity', 'temperature']

    # ct = ColumnTransformer([
    #     ('Scaler', StandardScaler(), col_names)
    # ], remainder='passthrough')

    # df[col_names] = ct.fit_transform(df[col_names])

    # df = create_date_features(df)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)

def main():
    output_path = INPUT_PATH.replace('raw', 'processed')
    for dir in os.listdir(INPUT_PATH):
        for file_name in os.listdir(os.path.join(INPUT_PATH, dir)):
            input_file_name = os.path.join(INPUT_PATH, dir, file_name)
            output_file_path = os.path.join(output_path, dir, file_name)
            process_data(input_file_name, output_file_path)

    # for file_name in tqdm(os.listdir(INPUT_PATH)):
    #     input_file_name = os.path.join(INPUT_PATH, file_name)
    #     output_file_path = os.path.join(output_path, file_name)
    #     process_data(input_file_name, output_file_path)

if __name__ == '__main__':
    main()

