import pickle
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Model
#set some placeholders for model parameters
NUM_LSTM = 200
NUM_DENSE_LSTM = [100, 50]
NUM_FILTERS = [128,32]
POOL_SIZE = 5
DROPOUT = .2
KERNEL_SIZES = [3,4,5]
#TOKENIZER = pickle.load(open('tokenizer', 'r'))

def rnn_cnn_concat_pretrained( num_words, embedding_dim, embedding_matrix, input_length, num_filters = NUM_FILTERS, 
                                pool_size = POOL_SIZE, num_lstm = NUM_LSTM, num_dense_list = NUM_DENSE_LSTM, dropout = .2 ):
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_length,
                            trainable=False)
    print(input_length)
    sequence_input = Input(shape = (input_length,), dtype='int32')
    embedded_sequences =embedding_layer(sequence_input)
    
    #defines the rnn portion of the model
    y = Bidirectional(LSTM(num_lstm, return_sequences = True))(embedded_sequences)
    y = Dropout(.2)(y)
    for d in num_dense_list:
        y = Dense(d, activation='relu')(y)
        y = Dropout(dropout)(y)
    
    y = Dense(1, activation = 'relu')(y)
    rnn_output = Flatten()(y)

    #now lets define the convolutional layer
    x = Conv1D(num_filters[0], 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(pool_size)(x)
    x = Conv1D(num_filters[1], 5, activation='relu')(x)
    cnn_output = Flatten()(x)
    z = concatenate([cnn_output,rnn_output])

    #now lets define our final dense layer
    z = Dense(32, activation='relu')(z)
    preds = Dense(1, activation = 'sigmoid')(z)
    
    #final model
    model = Model(sequence_input, preds)
    print(model.summary)
    return model

def rnn_pretrained(num_words, embedding_dim, embedding_matrix, input_length):
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_length,
                            trainable=False)
    
    sequence_input = Input(shape = (input_length,), dtype='int32')
    embedded_sequences =embedding_layer(sequence_input)

    y = Bidirectional(LSTM(NUM_LSTM))(embedded_sequences)
    y = Dropout(.2)(y)
    for d in NUM_DENSE_LSTM:
        y = Dense(d, activation='relu')(y)
        y = Dropout(DROPOUT)(y)
    
    y = Dense(1, activation = 'sigmoid')(y)
    model = Model(sequence_input, y)
    return model



def convolution(num_words, embedding_dim, embedding_matrix, input_length):
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_length,
                            trainable=False)
    
    sequence_input = Input(shape = (input_length,), dtype='int32')
    embedded_sequences =embedding_layer(sequence_input)
    
    first_conv = Conv1D(NUM_FILTERS[1], KERNEL_SIZES[0], padding='valid', activation='relu', strides=1)(embedded_sequences)
    first_pool = GlobalMaxPooling1D()(first_conv)

    second_conv = Conv1D(NUM_FILTERS[1], KERNEL_SIZES[1], padding='valid', activation='relu', strides=1)(embedded_sequences)
    second_pool = GlobalMaxPooling1D()(second_conv)

    third_conv = Conv1D(NUM_FILTERS[1], KERNEL_SIZES[2], padding='valid', activation='relu', strides=1)(embedded_sequences)
    third_pool = GlobalMaxPooling1D()(third_conv)

    z = concatenate([first_pool,second_pool, third_pool])

    z = Dense(48, activation='relu')(z)
    preds = Dense(1, activation = 'sigmoid')(z)

    model = Model(sequence_input, preds)
    
    return model