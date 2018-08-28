import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words= 10000)

#inputs to a neural network must be the same length, and currently it isnt
print(len(train_data[0]),len(train_data[1]))

#converting the integers to words

word_index = imdb.get_word_index()

#the first indice are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown value
word_index["<UNUSED>"] = 3  

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i,'?') for i in text])

#print(decode_review(train_data[23]))

#we can either use a one-hot encoding method to convert them to a vector 
# of same dimention or we could pad the array so that they all have the 
#same length and then create an integer tensor of shape num_example*max_length
# here we are using an embedding layer capable of handling this shape
# as the first layer of our nn

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


print(len(train_data[0]), len(train_data[1]))

#print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#Here we are using a gradient descent optimization
#introducting the compiling parameters
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#creating a validation training set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#evaluating the model on the test set
results = model.evaluate(test_data, test_labels)
print(results)

