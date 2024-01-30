from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pymysql

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
    wines = pd.read_csv('/Users/nick/Documents/GitHub/spingle-dingle/data/final_wine_data.csv') # change later
    wines['type_'] = wines['type_'].apply(lambda x : 0 if x=='red' else 1) # dummy code variables
    wines.drop(columns=["Unnamed: 0", "name_", "wine_index", "alcohol"], inplace=True)
    knn = NearestNeighbors(n_neighbors=neighbors, metric='euclidean')
    knn.fit(wines)
    return knn

def make_request():
    # set_up_knn()
    knn = set_up_knn()

    # Its important to use binary mode 
    knnPickle = open('/Users/nick/Documents/GitHub/spingle-dingle/data/knnpickle_file', 'wb') 
        
    # source, destination 
    pickle.dump(knn, knnPickle)  

    # close the file
    knnPickle.close()
