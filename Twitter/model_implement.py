from __future__ import print_function
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
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from models import rnn_cnn_concat_pretrained, rnn_pretrained, convolution
import os
from keras import metrics
from keras.models import Model


GLOVE_DIR = './glove.6B'
EMBEDDING_DIM = 100
#print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#print('Found %s word vectors.' % len(embeddings_index))


tweets = np.load('tweets_np_array_no_hashtags.npy')
labels_np = np.load('labels_np_array_no_hashtags.npy')

tokenizer= Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)
word_index = tokenizer.word_index
# prepare embedding matrix
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector





# print('Found %s unique tokens.' % len(word_index))
# print('new')

data = pad_sequences(sequences)



model = convolution(num_words, EMBEDDING_DIM ,embedding_matrix, data.shape[1])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [metrics.binary_accuracy])

#save the old model bc i forgot too last time
# model2 = rnn_cnn_concat_pretrained(num_words, EMBEDDING_DIM ,embedding_matrix, data.shape[1])
# with open('./weights/merged_model_senators_200_LSTM/model_summary.txt','w+') as fh:
#     model2.summary(print_fn=lambda x: fh.write(x + '\n'))

with open('./weights/simple_conv_model_no_hashtags/model_summary.txt','w+') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


filepath="./weights/simple_conv_model_no_hashtags/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience =3)
tensorboard = TensorBoard()
callbacks_list = [checkpoint, earlystop, tensorboard]
model.fit(data[:-1000], labels_np[:-1000], nb_epoch = 10, batch_size = 128, validation_split =.1, callbacks = callbacks_list)
scores = model.evaluate(data[-1000:], labels_np[-1000:])
print(scores)

np.save('senator_test_x_simple_conv_model_no_hashtags',data[:-1000])
np.save('senator_test_y_simple_conv_model_no_hashtags',labels_np[:-1000])