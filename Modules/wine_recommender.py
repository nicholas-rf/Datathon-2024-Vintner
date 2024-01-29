from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# SAMPLE 236       3.5     3.5          4       2.5        red
# SAMPLE 235         3       3          4         2        red
# SAMPLE 233         3       2          3         3        red

"""
This module contains the recommender system for the wine based off of a euclidian K-nearest-neighbors algorithm with cosine-similarity used to add variety to the recommendations. 
"""

#example data
wines = np.array([
    [5, 2, 4, 0, 3],
    [3, 5, 2, 5, 3],  
    [3, 5, 0, 5, 2]  
])

scalar = StandardScaler()
wines_scaled = scalar.fit_transform(wines)
knn.fit(wines_scaled)

user_preference = np.array([["prefs go here"]])
user_preferance_scaled = scalar.transform(user_preference)

distances, indices = knn.kneighbors(user_preferance_scaled)


def set_up_knn(neighbors=20):
    """
    Sets up a k-nearest-neighbors model to provide recommendations from for a session.
    
    Args:
        neighbors (int) : The number of recommendations to return via k-nearest-neighbors.
    
    Returns:
        knn (sklearn.neighbors.NearestNeighbors) : A knn model for recommendation gathering.
    """
    scalar = StandardScaler()
    wines = pd.read_csv('/Users/nick/Documents/GitHub/spingle-dingle/scripts/users.csv') # change later
    wines_scaled=scalar.fit_transform(wines) # Apply a scalar so that mean and std get normalized
    knn = NearestNeighbors(n_neighbors=neighbors, metric='euclidian')
    knn.fit(wines_scaled)
    return knn

def make_request():
    knn = set_up_knn()
    request_catalogue = lambda user_info : knn.kneighbors(user_info)
    wines = request_catalogue(user_info="info here")
    similarity_mat = cosine_similarity(wines)

    # now to figure out how to get a variety of cosine values, maybe use binning on the similarity vectors? 