from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle

# SAMPLE 236       3.5     3.5          4       2.5        red
# SAMPLE 235         3       3          4         2        red
# SAMPLE 233         3       2          3         3        red

"""
This module contains the recommender system for the wine based off of a euclidian K-nearest-neighbors algorithm with cosine-similarity used to add variety to the recommendations. 
"""

def set_up_knn(neighbors=5):
    """
    Sets up a k-nearest-neighbors model to provide recommendations from for a session.
    
    Args:
        neighbors (int) : The number of recommendations to return via k-nearest-neighbors.
    
    Returns:
        knn (sklearn.neighbors.NearestNeighbors) : A knn model for recommendation gathering.
    """
    # scalar = StandardScaler()
    wines = pd.read_csv('/Users/nick/Documents/GitHub/spingle-dingle/feature_data.csv') # change later
    wines['type'] = wines['type'].apply(lambda x : 0 if x=='red' else 1) # dummy code variables
    wines.drop(columns=["Unnamed: 0"], inplace=True)
    # wines_scaled=scalar.fit_transform(wines) # Apply a scalar so that mean and std get normalized
    knn = NearestNeighbors(n_neighbors=neighbors, metric='euclidean')
    knn.fit(wines)
    return knn, wines

def make_request():
    knn, wines = set_up_knn()
    feature_vectors = knn._fit_X 

    print(feature_vectors)
    # with open('/Users/nick/Documents/GitHub/spingle-dingle/data/knn_vectors.pk1', 'wb') as f:
    #     pickle.dump((feature_vectors), f)

    # _, indices = knn.kneighbors([[0, 6, 1, 4, 4, 0, 3]])
    # recommended_wines = wines.iloc[indices[0]]  # Use indices to retrieve wines
    # print(recommended_wines)
    # request_catalogue = lambda user_info : knn.kneighbors(user_info)
    # distance, indices = request_catalogue(user_info=[[0, 6, 1, 4, 4, 0, 3]])
    # similarity_mat = cosine_similarity(wines)

    # now to figure out how to get a variety of cosine values, maybe use binning on the similarity vectors? 

