# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
X_train = np.load('/home/arasu/FYP/code/sample/X_train.npy')
X_val = np.load('/home/arasu/FYP/code/sample/X_val.npy')
X_test = np.load('/home/arasu/FYP/code/sample/X_test.npy')

y_train = np.load('/home/arasu/FYP/code/sample/y_train.npy')
y_val = np.load('/home/arasu/FYP/code/sample/y_val.npy')
y_test = np.load('/home/arasu/FYP/code/sample/y_test.npy')


def normalize_it(X):
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X

from keras.utils import np_utils, generic_utils

X_train = normalize_it(X_train)
X_val = normalize_it(X_val)
X_test = normalize_it(X_test)

y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)
y_val = np_utils.to_categorical(y_val, 3)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)
X_val, y_val = shuffle(X_val, y_val, random_state=0)

X_train = np.expand_dims(X_train, axis=4)
X_val = np.expand_dims(X_val, axis=4)
X_test = np.expand_dims(X_test, axis=4)


from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Activation


model = Sequential()

model.add(Conv3D(16, (6, 6, 6), strides = 1, input_shape=(20, 100, 50, 1), activation='relu', padding='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.5))

model.add(Conv3D(8, (3, 3, 3), strides = 1, activation='relu', padding='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])



model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

ypred = model.predict(X_test)

ypred[:5]