"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""
import os
import torch
import numpy as np
import pandas as pd
import datetime as dt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class irisTrainSet(Dataset):
    def __init__(self):
        #load
        datatrain = pd.read_csv('../Datasets/iris/iris_train.csv')
        
        #change string value to numeric
        datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
        datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
        datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
        datatrain = datatrain.apply(pd.to_numeric)
        
        self.df = datatrain
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.df.iloc[idx, :4]
        label = self.df.iloc[idx, 4]

        return (np.array(features), label)

class irisTestSet(Dataset):
    def __init__(self):
        #load
        datatrain = pd.read_csv('../Datasets/iris/iris_test.csv')
        
        #change string value to numeric
        datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
        datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
        datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
        datatrain = datatrain.apply(pd.to_numeric)
        
        self.df = datatrain
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.df.iloc[idx, :4]
        label = self.df.iloc[idx, 4]

        return (np.array(features), label)
