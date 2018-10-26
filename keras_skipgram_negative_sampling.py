# -*- coding: utf-8 -*-
"""Keras Skipgram Negative Sampling

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fUhN1Z13-4lfEbPYkYzwyu9WrbgieLLL
"""

import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.models import Sequential, Model
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer

df = pd.read_csv('combined_datasets.csv', index_col=None)
NUM_SENTENCES = len(df['0'])
corpus = df['0'].astype(str)

f = lambda x: x.translate(None,'!"%&\'()*+,-./:;<=>?[\\]^_`{|}~')
corpus = corpus.apply(f)

corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
V = len(tokenizer.word_index) + 1

dim_embedddings = 24
model = Sequential()
model.add(Embedding(V, dim_embedddings, input_length=5))

dim_embedddings = 24

embedding = Embedding(V, dim_embedddings)

# inputs
w_inputs = Input(shape=(1,), dtype='int32')
w = embedding(w_inputs)

# context
c_inputs = Input(shape=(1,), dtype='int32')
c = embedding(c_inputs)
o = Dot(axes=2)([w, c])
o = Reshape((1,), input_shape=(1, 1))(o)
o = Activation('sigmoid')(o)

SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
SkipGram.summary()
SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

from math import ceil

batch_size = 256

for _ in range(5):
    loss = 0.
    for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):
        data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)
        x = [np.array(x).reshape(-1, 1) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        if x:
            for i in range(int(len(x) / batch_size)):
                x_batch = [x[0][i * batch_size:(i + 1) * batch_size], x[1][i * batch_size:(i + 1) * batch_size]]
                loss += SkipGram.train_on_batch(x_batch, y[i * batch_size:(i + 1) * batch_size])

    print(loss)

f = open('vectors_negative_sampling.txt', 'w')
f.write('{} {}\n'.format(V - 1, dim_embedddings))
vectors = SkipGram.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()
