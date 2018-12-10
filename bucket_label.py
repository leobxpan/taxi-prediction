from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pdb

def bucket_fare_dist(fare_dist_input_path, fare_dist_output_path, num_clusters_fare, num_clusters_distance):
    df = pd.read_pickle(fare_dist_input_path)
    
    kmeans_fare = KMeans(n_clusters=num_clusters_fare, n_jobs=-1).fit(df['fare_amount'].values.reshape(-1, 1))
    centroids_fare = kmeans_fare.cluster_centers_
    bucketed_fare = kmeans_fare.labels_ 
    
    kmeans_distance = KMeans(n_clusters=num_clusters_distance, n_jobs=-1).fit(df['trip_distance'].values.reshape(-1, 1))
    centroids_distance = kmeans_distance.cluster_centers_
    bucketed_distance = kmeans_distance.labels_
    
    # Add new columns to dataframe 
    df = df.assign(bucketed_fare=pd.Series(bucketed_fare, index=df.index))
    df = df.assign(bucketed_distance=pd.Series(bucketed_distance, index=df.index))
    
    df.to_pickle(fare_dist_output_path)

def bucket_activity(activity_input_path, activity_output_path, num_clusters_activity):
    df = pd.read_pickle(activity_input_path)
    kmeans_activity = KMeans(n_clusters=num_clusters_activity, n_jobs=-1).fit(df[0].values.reshape(-1, 1))
    centroids_activity = kmeans_activity.cluster_centers_
    bucketed_activity = kmeans_activity.labels_ 
    
    # Add new columns to dataframe 
    df = df.assign(bucketed_activity=pd.Series(bucketed_activity, index=df.index))
    
    df.to_pickle(activity_output_path)


fare_dist_input_path = '../assets/clean-data/merged_141516.pickle'
fare_dist_output_path = '../assets/clean-data/merged_bucketed_141516.pickle'
activity_input_path = '../assets/clean-data/activity_141516.pickle'
activity_output_path = '../assets/clean-data/activity_bucketed_141516.pickle'

num_clusters_activity = 10
num_clusters_fare = 10
num_clusters_distance = 10

#bucket_fare_dist(fare_dist_input_path, fare_dist_output_path, num_clusters_fare, num_clusters_distance)

bucket_activity(activity_input_path, activity_output_path, num_clusters_activity)


