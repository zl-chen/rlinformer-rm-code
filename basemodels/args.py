# %% [markdown]
# ## 1. 引入

# %%
import argparse
import datetime
import json
import os
import shutil
import sys
import pickle 
import time

start = time.time()
import pandas as pd
import torch

from utils.visualization import *
from utils.initialize_random_seed import *
from utils.metrics import *
from utils.multi_lag_processor import *
from pyecharts.globals import CurrentConfig, OnlineHostType
import warnings


import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

# np.set_printoptions(precision=2)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行


import matplotlib.pyplot as plt

def get_len(x):
    i = 0 
    for _ in x:
        i += 1
    return i

battery_df =pd.read_csv("../dataset/processed_data/shu_old_fade_rate.csv",index_col=0)

cols = list(battery_df.columns)

train_rate = 0.6
total_length = get_len(cols)
train_length = int(total_length * train_rate)

train_length = 22


random.seed(datetime.datetime.now())
train_all_cols = random.sample(cols,train_length)

# %%
import itertools

def sort_train_all_cols(v):
    
    return float(v.split("A")[0])

train_all_cols.sort(key=sort_train_all_cols)

if type(len) == int:
    del len


test_cols = [x for x in cols if x not in train_all_cols]

"""
train_all_cols = ['3A_(1)', '4A_(1)', '5A_2C_(1)', '6A_(1)', '7A_(1)', '7.5A_3C_(4)', '7.5A_3C_(3)', '8A_(3)', '9A_(3)', '10.5A_(2)', 
'10.75A_(2)', '10.75A_(3)', '11A_(3)', '12A_(2)', '12A_(3)', '12.5A_5C_(2)', 
'12.5A_5C_(3)', '13A_(1)', '14A_(2)', '15A_6C_(2)', '15A_6C_(1)', '15A_6C_(3)']

test_cols = ['1A_(1)', '2A_(1)', '2.5A_1C_(2)', '2.5A_1C_(3)',
'2.5A_1C_(1)', '3A_(2)', '5A_2C_(3)', '5A_2C_(4)', '7.5A_3C_(1)', 
'8A_(1)', '9A_(1)', '10A_(2)', '11A_(4)', '12.5A_5C_(1)', '14A_(1)']

"""

import random  
random.shuffle(train_all_cols)
random.shuffle(test_cols)

my_train_args = (train_all_cols,test_cols)

print(my_train_args)

with open("./args/my_train_args.pkl","wb") as f:
    pickle.dump(my_train_args,f)
    
    