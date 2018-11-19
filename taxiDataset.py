class taxiDataset(Dataset):
    """Taxi Data dataset."""
    
    def __init__(self, csv_file, root_dir, desired_labels, split):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        PU_dt = 'tpep_pickup_datetime'
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df_all = pd.read_csv(os.path.join('/home/lashi/assets', csv_file),usecols=np.arange(17),parse_dates=[PU_dt],date_parser=dateparse)
        self.dayDict = self._createDayDict()
        self.timeBuckets = self._createTimeBuckets()
        self.df = self._parsetime(df_all, desired_labels, PU_dt, split)
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        PU_loc = self.df.iloc[idx, 0]
        PU_dow = self.dayDict[self.df.iloc[idx, 1]]
        PU_tB = self.timeBuckets[self.df.iloc[idx, 2]]
        PU_num = self.df.iloc[idx, 3]
        sample = {'PU_loc': PU_loc, 'PU_dow': PU_dow, 'PU_tB': PU_tB, 'PU_num': PU_num}

        return sample
    
    def _createDayDict(self):
        # loop to create day of week dictionary
        day_dict = dict()
        for i in range(7):
            oneHotVec = np.zeros((1,7)).astype('int')
            oneHotVec[0,i] = 1
            day_dict[i] = oneHotVec
        
        return day_dict
    
    def _createTimeBuckets(self):
        # loop to create time bucket dictionary
        time_buckets = dict()
        split = 30
        numBuckets = int(24*(60/split))
        ind = 0
        for hr in range(24):
            for i in range(int(60/split)):
                key = '%02d'%hr+':'+'%02d'%(i*split)+':00'
                oneHotVec = np.zeros((1,numBuckets)).astype('int')
                oneHotVec[0,ind] = 1
                time_buckets[key] = ind
                time_buckets[ind] = oneHotVec
                ind += 1
                
        return time_buckets
    
    def _parsetime(self, data_frame, desired_labels, dt_label, split):
        data_frame[dt_label] = pd.to_datetime(data_frame[dt_label])
        data_frame[dt_label] = data_frame[dt_label].dt.round('30min')
        data_frame[desired_labels[1]] = data_frame[dt_label].dt.dayofweek
        data_frame[desired_labels[2]] = data_frame[dt_label].dt.time.apply(lambda x: self.timeBuckets[str(x)])
        
        return data_frame.groupby(desired_labels).size().reset_index()
