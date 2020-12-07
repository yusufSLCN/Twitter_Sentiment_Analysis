from tensorflow import keras
from sentimentPredictor import preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
import numpy as np

testData = ['Nothing goes the way I want', "I cant wait to see the results", "not so bad I guess",
            'I dont feel good', 'wow thats interesting', 'Ohh it is monday again!!',
            'the service time was too long', 'I dont think you are a bad leader']

testData = [preprocess(sent, stem=True) for sent in testData]
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 300
vocab_size = 10000
seqs = tokenizer.texts_to_sequences(testData)
padded = pad_sequences(seqs, maxlen=max_length, padding='post', truncating='post')

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model('./models/sentPred_embedDim32_vocab100000')
print(model.predict(padded))