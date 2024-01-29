# Hugging face
from transformers import pipeline

# Pandas
import pandas as pd

"""
This module contains code to apply the BERT review score model onto the dataset containing unscored reviews.
"""

review_assigner = pipeline("sentiment-analysis", model='../Datathon/wine_review_prediction_model')

# then from here we could just assign a column with all the reviews

unscored_reviews = pd.read_csv("")
