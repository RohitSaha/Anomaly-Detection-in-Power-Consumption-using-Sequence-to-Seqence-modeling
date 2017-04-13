import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import glob
import time

#Loading training data.
data_X = np.zeros((1, 6), 'float')
data_Y = np.zeros((1, 19), 'float')
training_data = glob.glob('training_data.npz')
for i in training_data:
    with np.load(i) as data:
        #print data.files
        training_temp = data['train_X']
        labels_temp = data['train_Y']
    data_X = np.vstack((data_X, training_temp))
    data_Y = np.vstack((data_Y, labels_temp))
data_X = data_X[1:, :]
data_Y = data_Y[1:, :]
print (data_X,data_X.shape)
print (data_Y,data_Y.shape)
seq_length = 6

#############################################################################
############################# CLASS LABEL ###################################
#############################################################################

#0-25, 25-50, 50-75, 75-100, 100-150, 150-200, 200-300, 300-400
#400-500, 500-600, 600-700, 700-800, 800-900, 900-1000, 1000-2000
#2000-3000, 3000-4000, 4000-5000, 5000-6000
#19 classes

dict = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
        '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
        '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
        '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1,
        '4001-5000' : 1, '5001-6000' : 1}

n_patterns = len(data_X)
print ("Number of data entries", n_patterns)
#Arranging the data in a form that can be provided to the Deep Learning architecture using Keras.
X = np.reshape(data_X, (n_patterns, seq_length, 1))
Y = data_Y

print ("Data prepared")

#Creating the Keras model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.7))
model.add(LSTM(256))
model.add(Dropout(0.7))
model.add(Dense(Y.shape[1], activation='softmax'))

#sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
filename = "weights-improvement.hdf5"
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer=rms)

filepath = "weights_improvement2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tfboard=keras.callbacks.TensorBoard(log_dir='tfboard', histogram_freq=1, write_graph=True, write_images=False)
callbacks_list = [checkpoint,tfboard]

print ("Training started.....")
start = time.time()
model.fit(X, Y, nb_epoch=200, batch_size=128, callbacks=callbacks_list, verbose=1)
end = time.time()
print ("Training complete")
print ("Time taken to train : ", (end - start))