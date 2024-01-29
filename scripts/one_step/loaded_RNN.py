import numpy as np
import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
import os
import time

one_step_loaded_red = tf.saved_model.load('one_step_red')
one_step_loaded_white = tf.saved_model.load('one_step_white')


#Generates red wine names:


# for n in range(1599):
#     next_char, states = one_step_loaded_red.generate_one_step(next_char, states = states)
#     result.append(next_char)

# result = tf.strings.join(result)
# print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)

def add_name(color):
    """
    Creates a name for a wine.

    Args:
        color (str) : A color of the wine to determine which RNN model to use.
    
    Returns:
        Wine name (str) : A wine name 
    """
    states = None
    next_char = tf.constant(['M'])
    result = [next_char]
    name = []   

    #for i in range(15):
    # Our color is red, therefore we generate from the red model
    if color == 'red':
        for n in range(50):
            next_char, states = one_step_loaded_red.generate_one_step(next_char, states = states)
            result.append(next_char)
        name.append(tf.strings.join(result)[0].numpy().decode('utf-8').split('.')[0])
    else:
        for n in range(50): 
            next_char, states = one_step_loaded_white.generate_one_step(next_char, states = states)
            result.append(next_char)
        name.append(tf.strings.join(result)[0].numpy().decode('utf-8').split('.')[0])
    
    return name
    
print(add_name("red"))
print(add_name("white"))



#Generates white wine names:
#states = None
#next_char = tf.constant(['13th'])
#result = [next_char]

#for n in range(300):
    #next_char, states = one_step_loaded_white.generate_one_step(next_char, states = states)
    #result.append(next_char)

#result = tf.strings.join(result)
#print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)