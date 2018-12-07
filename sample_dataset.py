import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import gc
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

csv_path = csv_path_lst_2014[0]
print('Reading first file:',csv_path)
df = pd.read_csv('/home/lashi/assets/test_40.csv',usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
scrap, df = train_test_split(df_temp,test_size=0.01)
df.to_pickle('/home/lashi/assets/sampled_data_2014.pickle')
del df, scrap

print(len(scrap))

# for i, csv_path in enumerate(csv_path_lst_2014):
#     if i != 0:
#         print(i,'Reading',csv_path)
#         df_temp = pd.read_csv(csv_path,usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
#         print('Sampling...')
#         scrap, df_temp = train_test_split(df_temp,test_size=0.01)
#         print(len(df_temp))
#         df.append(df_temp)
#         del df_temp
#         gc.collect
#     if i == 1:
#         break



