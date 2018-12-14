import os
import torch
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import gc
import random
from sklearn.model_selection import train_test_split
import pdb

years = ['2014','2015','2016']
months = list(range(1,13))
time_suffix_lst = []
for yr in years:
    for month in months:
        time_suffix_lst.append(yr+'-'+'%02d'%month)
time_suffix_lst_2014 = time_suffix_lst[6:12]
time_suffix_lst = time_suffix_lst[12:-6]

csv_file_lst_2014 = ['yellow_tripdata_' + time_suffix + '.csv' for time_suffix in time_suffix_lst_2014]
csv_path_lst_2014 = [os.path.join('/home/lashi/assets', csv_file) for csv_file in csv_file_lst_2014]
csv_file_lst = ['yellow_tripdata_' + time_suffix + '.csv' for time_suffix in time_suffix_lst]
csv_path_lst = [os.path.join('/home/lashi/assets', csv_file) for csv_file in csv_file_lst]

# Save data as pickle files
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
PU_dt = 'tpep_pickup_datetime'

for i, csv_path in enumerate(csv_path_lst):
    print('------------------------------------------------')
    print('Reading',csv_path)
    df = pd.read_csv(csv_path,usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
    print('Done')
    df_pickle = csv_path[:-3] + 'pickle'
    print('Saving',df_pickle)
    print(df_pickle)
    df.to_pickle(df_pickle)
    del df
