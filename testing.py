import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
import glob
from keras import optimizers

#Loading training data.
data_X = np.zeros((1, 6), 'float')
data_Y = np.zeros((1, 19), 'float')
training_data = glob.glob('training_data/*.npz')
for i in training_data:
    with np.load(i) as data:
        #print data.files
        training_temp = data['train_X']
        labels_temp = data['train_Y']
    data_X = np.vstack((data_X, training_temp))
    data_Y = np.vstack((data_Y, labels_temp))


data_X = data_X[1:, :]
data_Y = data_Y[1:, :]
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
print "Number of data entries", n_patterns
#Arranging the data in a form that can be provided to the Deep Learning architecture using Keras.
X = np.reshape(data_X, (n_patterns, seq_length, 1))
Y = data_Y

print "Data prepared"

#Creating the Keras model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.7))
model.add(LSTM(256))
model.add(Dropout(0.7))
model.add(Dense(Y.shape[1], activation='softmax'))
'''
filename = "weights-improvement=381-0.971429.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
'''
sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.0, nesterov=False)

filename = "mt-weights-improvement=10-0.023081.hdf5"
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer=sgd)

print "Model created....."
print "Testing....."

#For testing
checking = {0:'0-25', 1:'26-50', 2:'51-75', 3:'76-100', 4:'101-150', 5:'151-200',
            6:'201-300', 7:'301-400', 8:'401-500', 9:'501-600', 10:'601-700',
            11:'701-800', 12:'801-900', 13:'901-1000', 14:'1001-2000', 15:'2001-3000',
            16:'3001-4000', 17:'4001-5000', 18:'5001-6000'}

accurate = 0

prediction = model.predict(X, verbose=0)
index = prediction.argmax(-1)
true_labels = Y.argmax(-1)
print index[0:25]
print "---------------------------"
print true_labels[0:25]
print "Training data"
print X[0:25]*5853.0

accuracy = np.mean(index == true_labels)
print "Accuracy : ", accuracy*100, "%"


'''
#Testing 1 by 1.
for i in range(0, n_patterns):
    pattern = data_X[i]
    test_X = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(test_X, verbose=0)[0]
    index = np.argmax(prediction)
    if (i > 499) and (i % 500 == 0):
        print i, " data points tested."
    #print checking[index], np.argmax(data_Y[i])
    #print index, np.argmax(data_Y[i])

    if index == np.argmax(data_Y[i]):
        accurate += 1

print "Accuracy : ", (accurate/float(n_patterns))*100
'''
'''
Log :
Time taken to train :  27438.9335561

1st training : (using adam and Cross_Entropy
77.39% using 319
78.77% using 334
80.9698% using 393

2nd training : (using SGD and M.S.E)
Loss not changing after 19th epoch
81.55% using 05
81.58% using 10

'''
