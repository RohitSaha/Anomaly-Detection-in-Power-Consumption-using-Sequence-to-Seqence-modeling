import pickle
import time
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import keras


#Opening the file
with open("oneArrayPowers", 'rb') as file:
    unroll = pickle.load(file)
print(unroll)
#############################################################################
#############################DATA PRE-PROCESSING#############################
#############################################################################
# with np.load('training_data.npz') as data:
#     train_X=data['train_X']
#     train_Y=data['train_Y']
# print(train_X.shape)
# print(train_X[0])
# print(train_Y.shape)
# print(train_Y[0])

#Converting the list elements to integers
for i in range(len(unroll)):
    unroll[i] = int(unroll[i])
#
# data_48 = []
# for i in range(0, len(data), 48):
#     num = (data[i: i+48])
#     data_48.append(num)
#
# Removing data points which have 0's.
# data_48_clean = []
# for i in data_48:
#     if 0 not in i:
#         data_48_clean.append(i)
#
# Unrolling all the data points into a single row vector
# unroll = []
# for row in data_48_clean:
#     for values in row:
#         unroll.append(values)

#Keeping track of maximum value for normalizing train_X.
max_value = max(unroll)
print ("max_value ", max_value)
############################### END ##########################################

#############################################################################
############################# CLASS LABEL ###################################
#############################################################################

def no_classes():
    sortUnroll = np.sort(unroll)
    num_classes = 8
    perClass = int(len(sortUnroll)/num_classes)
    c = []
    for i in range(0,len(sortUnroll),perClass):
        c.append(sortUnroll[i:i+perClass])
    intervals = []
    for i in c:
        intervals.append(max(i))
    print (intervals)

# no_classes()

#0-25, 25-50, 50-75, 75-100, 100-150, 150-200, 200-300, 300-400
#400-500, 500-600, 600-700, 700-800, 800-900, 900-1000, 1000-2000
#2000-3000, 3000-4000, 4000-5000, 5000-6000
#19 classes

# dict = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
#         '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
#         '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
#         '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1,
#         '4001-5000' : 1, '5001-6000' : 1}

dict = {'1-45':1, '46-82':1, '83-132':1, '133-197':1, '198-309':1, '310-584':1, '585-705':1}

def calculate_1_hot(value):
    hot_1_vector = []
    for i in dict.keys():
        values = i.split('-')
        if value >= int(values[0]) and value <= int(values[1]):
            hot_1_vector.append(1)
        else:
            hot_1_vector.append(0)
    return hot_1_vector


def normalize_train_X(data):
     return data/max_value

#Arranging the data in a proper format
seq_length = 6
train_X = np.zeros((1, 6), 'float')
train_Y = np.zeros((1, 7), 'float')
for i in range(0, len(unroll) - seq_length, 1):
    if i%100000 == 0:
        print(i,"/",len(unroll))
    train_X = np.vstack((train_X, unroll[i: i+seq_length]))
    get_1_hot_vector = calculate_1_hot(unroll[i+seq_length])
    train_Y = np.vstack((train_Y, get_1_hot_vector))

train_X = train_X[1:, :]
train_Y = train_Y[1:, :]

n_patterns = len(train_X)
train_X = normalize_train_X(train_X)
print("data normalized")
#Saving the training files to save load-up time.
np.savez('/output/training_data.npz', train_X=train_X, train_Y=train_Y)
print ("Training data saved.")

#Arranging the data in a form that can be provided to the Deep Learning architecture using Keras.
X = np.reshape(train_X, (n_patterns, seq_length, 1))
Y = train_Y

#Creating the Keras model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.7))
model.add(LSTM(256))
model.add(Dropout(0.7))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "/output/weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tfboard=keras.callbacks.TensorBoard(log_dir='/output/tfboard', histogram_freq=1, write_graph=True, write_images=False)
callbacks_list = [checkpoint,tfboard]

print ("Training started.....")
start = time.time()
model.fit(X, Y, nb_epoch=400, batch_size=128, callbacks=callbacks_list, verbose=1)
end = time.time()
print ("Training complete")
print ("Time taken to train : ", (end - start), "ms")