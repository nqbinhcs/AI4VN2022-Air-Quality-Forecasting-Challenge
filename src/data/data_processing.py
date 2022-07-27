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
sns.set()
warnings.filterwarnings(action='ignore')

INPUT_PATH = 'data/raw/data-train/input'

def process_data(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    df = df.iloc[:, 1:]

    df['PM2.5'].fillna(value = df['PM2.5'].mean(), inplace = True)
    df['humidity'].fillna(value=df['humidity'].mean(), inplace=True)
    df['temperature'].fillna(value=df['temperature'].mean(), inplace=True)

    # scale if fitting to regression
    # col_names = ['humidity', 'temperature']

    # ct = ColumnTransformer([
    #     ('Scaler', StandardScaler(), col_names)
    # ], remainder='passthrough')

    # df[col_names] = ct.fit_transform(df[col_names])

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)

def main():
    output_path = INPUT_PATH.replace('raw', 'processed')
    for file_name in os.listdir(INPUT_PATH):
        input_file_name = os.path.join(INPUT_PATH, file_name)
        output_file_path = os.path.join(output_path, file_name)
        process_data(input_file_name, output_file_path)

if __name__ == '__main__':
    main()

