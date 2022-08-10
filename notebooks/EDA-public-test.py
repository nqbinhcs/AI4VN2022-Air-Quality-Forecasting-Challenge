#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
warnings.filterwarnings(action='ignore')


# In[14]:


DATA_PATH = 'data/processed_interpolate/public-test/input'


# In[15]:


def count_null_value(file_name):
    df = pd.read_csv(file_name)
    na_25 = df['PM2.5'].isnull().sum() / len(df['PM2.5']) * 100
    # na_25 = len(df.iloc[list(df['PM2.5'].isnull() &
    #             df['humidity'].notnull() & df['temperature'].notnull())])
    return na_25


# In[16]:


for subfolder in os.listdir(DATA_PATH):
    dir_path = os.path.join(DATA_PATH, subfolder)
    print(f"==>", subfolder)
    for file_name in os.listdir(dir_path):
        print(file_name, ', Na value: ', count_null_value(os.path.join(dir_path, file_name)))


# In[ ]:




