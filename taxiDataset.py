import os
import torch
import numpy as np
import pandas as pd
import datetime as dt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pdb

class taxiDataset(Dataset):
    """Taxi Data dataset."""
    
    def __init__(self, csv_file=None, root_dir=None, df_file=None, desired_labels=None, slot=30, length=None, split='all', data_frame=None, time_range='year', pcs_per_month=4000):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        PU_dt = 'tpep_pickup_datetime'
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        
        self.dayDict = self._createDayDict()
        self.timeBuckets = self._createTimeBuckets(slot)
        self.df = None
        self.train = None
        self.val = None
        self.test = None
        self.root_dir = root_dir

        if df_file is not None:
            self.df = pd.read_csv(df_file)
            self.train = taxiDataset(split='train',data_frame=self.df)
            self.val = taxiDataset(split='val',data_frame=self.df)
            self.test = taxiDataset(split='test',data_frame=self.df)  

        if split == 'all' and df_file is None:
            if time_range == 'year':                                                                            # Sample equal amount of data from each month of a year
                time_suffix_lst = ['2016-07', '2016-08', '2016-09', '2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06']
                csv_file_lst = ['yellow_tripdata_' + time_suffix + '.csv' for time_suffix in time_suffix_lst]   # Someone else creates this smelly folder name, not me. ---- Leo 
                csv_path_lst = [os.path.join('/home/lashi/assets', csv_file) for csv_file in csv_file_lst]
                df_lst = []
                for csv_path in csv_path_lst:
                    df = pd.read_csv(csv_path,usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
                    df = self._parsetime(df, desired_labels, PU_dt)
                    df = df.sample(frac=1).reset_index(drop=True)[:pcs_per_month]
                    df_lst.append(df)
                self.df = pd.concat(df_lst)
                self.df = self.df.sample(frac=1).reset_index(drop=True)
                self.df.to_csv('./year_df.csv')
                self.train = taxiDataset(split='train',data_frame=self.df)
                self.val = taxiDataset(split='val',data_frame=self.df)
                self.test = taxiDataset(split='test',data_frame=self.df)  
            elif time_range == 'month':
                csv_path = os.path.join('/home/lashi/assets', csv_file)
                df_all = pd.read_csv(csv_path,usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
                df_all = self._parsetime(df_all, desired_labels, PU_dt)
        
                if length == None:
                    length = len(df_all)

                self.df = df_all.sample(frac=1).reset_index(drop=True)[:length]
                self.train = taxiDataset(split='train',data_frame=self.df)
                self.val = taxiDataset(split='val',data_frame=self.df)
                self.test = taxiDataset(split='test',data_frame=self.df)  
            else:
                raise ValueError("Unsupported time range. Try 'year' or 'month'.")
        elif split == 'test_only' and df_file is None:
            self.df = data_frame
        elif df_file is None:
            df_len = len(data_frame)
            train_ind = int(0.9*df_len)
            val_ind = int(0.5*(df_len-train_ind))+train_ind
            if split == 'train':
                self.df = data_frame[:train_ind]
            elif split == 'val':
                self.df = data_frame[train_ind:val_ind]
            elif split == 'test':
                self.df = data_frame[val_ind:]


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        PU_loc = self.df.iloc[idx, 1]
        PU_dow = self.df.iloc[idx, 2]
        PU_tB = self.df.iloc[idx, 3]
        # PU_dow = self.dayDict[self.df.iloc[idx, 1]]
        # PU_tB = self.timeBuckets[self.df.iloc[idx, 2]]
        PU_month = self.df.iloc[idx, 4]
        PU_num = self.df.iloc[idx, 5]
        # sample = {'PU_loc': PU_loc, 'PU_dow': PU_dow, 'PU_tB': PU_tB, 'PU_num': PU_num}
        sample = (np.array((PU_loc,PU_dow,PU_tB,PU_month)),PU_num)	

        return sample
   
    def _createDayDict(self):
        # loop to create day of week dictionary
        day_dict = dict()
        for i in range(7):
            oneHotVec = np.zeros((1,7)).astype('int')
            oneHotVec[0,i] = 1
            day_dict[i] = oneHotVec
        
        return day_dict
    
    def _createTimeBuckets(self, slot):
        # loop to create time bucket dictionary
        time_buckets = dict()
        numBuckets = int(24*(60/slot))
        ind = 0
        for hr in range(24):
            for i in range(int(60/slot)):
                key = '%02d'%hr+':'+'%02d'%(i*slot)+':00'
                oneHotVec = np.zeros((1,numBuckets)).astype('int')
                oneHotVec[0,ind] = 1
                time_buckets[key] = ind
                time_buckets[ind] = oneHotVec
                ind += 1
                
        return time_buckets
    
    def _parsetime(self, data_frame, desired_labels, dt_label):
        data_frame[dt_label] = pd.to_datetime(data_frame[dt_label])
        data_frame[dt_label] = data_frame[dt_label].dt.round('30min')
        data_frame[desired_labels[1]] = data_frame[dt_label].dt.dayofweek
        data_frame[desired_labels[2]] = data_frame[dt_label].dt.time.apply(lambda x: self.timeBuckets[str(x)])
        data_frame[desired_labels[3]] = data_frame[dt_label].dt.month

        return data_frame.groupby(desired_labels).size().reset_index()
