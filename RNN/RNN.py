import numpy as np
import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers
import os
import time

name_data = pd.read_csv('white_names.csv')
text = name_data['name'].tolist()
text = ". ".join(text)
vocab = sorted(set(text))

example_text = ['what', 'up']
chars = tf.strings.unicode_split(example_text, input_encoding = 'UTF-8')

ids_from_chars = keras.layers.StringLookup(vocabulary = list(vocab), mask_token=None)
ids = ids_from_chars(chars)
print(ids)

chars_from_ids = keras.layers.StringLookup(vocabulary = ids_from_chars.get_vocabulary(), invert = True, mask_token = None)
chars = chars_from_ids(ids)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis = -1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
print(all_ids)

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(12):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100
sequences = ids_dataset.batch(seq_length + 1, drop_remainder = True)
for seq in sequences.take(1):
    print(chars_from_ids(seq))

for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())

def split_input_target(sequence):
    """
    Takes in a sequence then duplicates and shifts it 
    to align the input and label for each timestep

    For training the model, we will need (input, label)
    pairs, where input and label are sequences

    Input is the current character, and label is the next
    at each timestep
    """
    input_text = sequence[: -1]
    target_text = sequence[1:]
    return input_text, target_text

print(split_input_target(list('whatup')))

dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print('Input: ', text_from_ids(input_example).numpy())
    print('Target: ', text_from_ids(target_example).numpy())

#Creating training batches:
#Here we will shuffle the data and then pack it into batches
batch_size = 69
buffer_size = 1000

dataset = (dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)
           .prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)

#Building the model:
vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = 256
rnn_units = 1024

class RNN(keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(rnn_units, return_sequences = True, return_state = True)
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, states = None, return_state = False, training = False):
        x = inputs
        x = self.embedding(x, training = training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training = training)
        x = self.dense(x, training = training)

        if return_state:
            return x, states
        else:
            return x
        
model = RNN(vocab_size = vocab_size, embedding_dim = embedding_dim, rnn_units = rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples = 1)
sampled_indices = tf.squeeze(sampled_indices, axis = -1).numpy()
print(sampled_indices)

#Creating optimizer and loss functions:
loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print('Prediction shape: ', example_batch_predictions.shape)
print('Mean loss: ', example_batch_mean_loss)

#Cheking to see if the exponential of mean loss is approximately equal to the vocab size:
print(tf.exp(example_batch_mean_loss).numpy())
print(len(vocab))

model.compile(optimizer = 'adam', loss = loss)
checkpoint_dir = './training_checkpionts'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = checkpoint_prefix,
                                                      save_weights_only = True)

epochs = 20

history = model.fit(dataset, epochs = epochs, callbacks = [checkpoint_callback])

class OneStep(keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature = 1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values = [-float('inf')] * len(skip_ids), indices = skip_ids,
                                      dense_shape = [len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    
    @tf.function
    def generate_one_step(self, inputs, states = None):
        #Converts strings to token ids:
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs = input_ids, states = states,
                                              return_state = True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples = 1)
        predicted_ids = tf.squeeze(predicted_ids, axis = -1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states
    

one_step_model_white = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['13th'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model_white.generate_one_step(next_char, states = states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time: ', end - start)


tf.saved_model.save(one_step_model_white, 'one_step_white')
one_step_loaded_white = tf.saved_model.load('one_step_white')