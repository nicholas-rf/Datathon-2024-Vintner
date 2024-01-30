import numpy as np
import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
import os
import time

"""
This module contains methods for generating either red or white wine names.
"""


# Load in our models for red and white wine names
one_step_loaded_red = tf.saved_model.load('one_step_red')
one_step_loaded_white = tf.saved_model.load('one_step_white')


def add_name(color):
    """
    Creates a name for a wine dependant on its color.

    Args:
        color (str) : A color of the wine to determine which RNN model to use.
    
    Returns:
        name (str) : A wine name 
    """

    # Initialize the states and the starting character
    states = None
    next_char = tf.constant(['M'])
    result = [next_char]
    name = []   

    # If our color is red, we will generate a name from the red names model
    if color == 'red':

        # Run the RNN 50 times and append each character to a result, then append the first name from the result
        for n in range(50):
            next_char, states = one_step_loaded_red.generate_one_step(next_char, states = states)
            result.append(next_char)
        name.append(tf.strings.join(result)[0].numpy().decode('utf-8').split('.')[0])
    
    # Our color is not red, therefore it is white
    else:
        for n in range(50): 
            next_char, states = one_step_loaded_white.generate_one_step(next_char, states = states)
            result.append(next_char)
        name.append(tf.strings.join(result)[0].numpy().decode('utf-8').split('.')[0])
    
    # Return the name
    return name
    
print(add_name("red"))
print(add_name("white"))