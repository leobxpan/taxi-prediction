import numpy as np
import pandas as pd
import pdb
from sklearn.cross_validation import train_test_split

pickle_path = '../assets/sampled_data_2014_clean.pickle'
time_slots = 30
PU_dt = 'pickup_datetime' 
desired_labels = ['pickup_id', 'day_of_week', 't_bucket', 'month_of_year']
seq_len = 5
feature_num = 4
train_per = .9
test_per = .05

def createTimeBuckets(slot):
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

def parsetime(data_frame, desired_labels, dt_label, timeBuckets):
    data_frame[dt_label] = pd.to_datetime(data_frame[dt_label])
    data_frame[dt_label] = data_frame[dt_label].dt.round('30min')
    data_frame[desired_labels[1]] = data_frame[dt_label].dt.dayofweek
    data_frame[desired_labels[2]] = data_frame[dt_label].dt.time.apply(lambda x: timeBuckets[str(x)])
    data_frame[desired_labels[3]] = data_frame[dt_label].dt.month

    return data_frame.groupby(desired_labels).size().reset_index()

def arr2seq(arr, seq_len, feature_num):
    
    """This function converts a 2-D array containing all data pieces into a 3-D matrix containing all X (which is a sequence) and a 1-D array of corresponding Y (which is a single number)."""
     
    X = []
    y = []

    for i in range(arr.shape[0] - feature_num - 1):
        if arr[i, 0] == arr[i+1, 0]:                                        # Two neighboring pieces are of the same location
            if arr[i, 0] == arr[i+seq_len, 0]:
                X.append(arr[i:i+seq_len, :feature_num])
                y.append(arr[i+seq_len, feature_num])
    
    return np.array(X), np.array(y)

df = pd.read_pickle(pickle_path)
df.columns = df.columns.str.lstrip()                                        # Remove the beginning space in column names
timeBuckets = createTimeBuckets(time_slots)

df = parsetime(df, desired_labels, PU_dt, timeBuckets)
df.sort_values(by=['pickup_id', 'month_of_year', 'day_of_week', 't_bucket'], inplace=True, ascending=True)

arr = df.values

X, y = arr2seq(arr, seq_len, feature_num)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_per))
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=(test_per/(1-train_per)))

np.save('../assets/X_train.npy', X_train)
np.save('../assets/X_val.npy', X_val)
np.save('../assets/X_test.npy', X_test)
np.save('../assets/y_train.npy', y_train)
np.save('../assets/y_val.npy', y_val)
np.save('../assets/y_test.npy', y_test)
