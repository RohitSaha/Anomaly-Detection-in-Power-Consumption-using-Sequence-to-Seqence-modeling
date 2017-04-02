import numpy as np
import pickle

#Opening the file
with open("training_data/Locality2.pickle", 'rb') as file:
    data = pickle.load(file)

#############################################################################
#############################DATA PRE-PROCESSING#############################
#############################################################################

#Converting the list elements to integers
for i in range(len(data)):
    data[i] = int(data[i])

data_48 = []
for i in range(0, len(data), 48):
    num = (data[i: i+48])
    data_48.append(num)

#Removing data points which have 0's.
data_48_clean = []
for i in data_48:
    if ((0 not in i) and (max(i)<6001)):
        data_48_clean.append(i)

#Unrolling all the data points into a single row vector
unroll = []
for row in data_48_clean:
    for values in row:
        unroll.append(values)

#Keeping track of maximum value for normalizing train_X.
max_value = max(unroll)
print "max_value ", max_value
############################### END ##########################################

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
        '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1}#,
     #   '4001-5000' : 1, '5001-6000' : 1}


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

# Arranging the data in a proper format
seq_length = 6
train_X = np.zeros((1, 6), 'float')
train_Y = np.zeros((1, 17), 'float')
for i in range(0, len(unroll) - seq_length, 1):
    train_X = np.vstack((train_X, unroll[i: i+seq_length]))
    get_1_hot_vector = calculate_1_hot(unroll[i+seq_length])
    train_Y = np.vstack((train_Y, get_1_hot_vector))

x=int(0.8*len(train_X))

test_X=train_X[x:,:]
test_Y=train_Y[x:,:]
train_X = train_X[1:x, :]
train_Y = train_Y[1:x, :]


n_patterns = len(train_X)
train_X = normalize_train_X(train_X)
test_X=normalize_train_X(test_X)
print "Number : ", n_patterns

#Saving the training files to save load-up time.
np.savez('locality2_trainingdata.npz', train_X=train_X, train_Y=train_Y)
np.savez('locality2_testdata.npz', train_X=test_X, train_Y=test_Y)
print "Training and testing data saved."
