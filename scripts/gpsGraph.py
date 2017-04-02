import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
import glob
from keras import optimizers
import matplotlib.pyplot as plt
import random



def calc(folder,out_nodes,max,start):
    #Loading training data.
    real_val_temp=[]
    real_val=[]
    pred_val_temp=[]
    pred_val=[]
    upper_bound=[]
    lower_bound=[]
    support={0:[0,25],1:[26,50],2:[51,75],3:[76,100],4:[101,150],5:[151,200],6:[201,300],7:[301,400],8:[401,500],9:[501,600],10:[601,700],11:[701,800],12:[801,900],13:[901,1000],14:[1001,2000],15:[2001,3000],16:[3001,4000],17:[4001,5000],18:[5001,6000]}



    data_X = np.zeros((1, 6), 'float')
    data_Y = np.zeros((1, out_nodes), 'float')
    training_data = glob.glob(folder+'/locality_trainingdata.npz')
    for i in training_data:
        with np.load(i) as data:
            #print data.files
            training_temp = data['train_X']
            labels_temp = data['train_Y']
        data_X = np.vstack((data_X, training_temp))
        data_Y = np.vstack((data_Y, labels_temp))


    data_X = data_X[start:, :]
    data_Y = data_Y[start:, :]
    seq_length = 6


    #############################################################################
    ############################# CLASS LABEL ###################################
    #############################################################################


    #0-25, 25-50, 50-75, 75-100, 100-150, 150-200, 200-300, 300-400
    #400-500, 500-600, 600-700, 700-800, 800-900, 900-1000, 1000-2000
    #2000-3000, 3000-4000, 4000-5000, 5000-6000
    #19 classes

    if(out_nodes==17):
        dict = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
                '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
                '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
                '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1}#,
               # '4001-5000' : 1, '5001-6000' : 1}
    else:
        dict = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
                '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
                '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
                '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1,
                '4001-5000' : 1, '5001-6000' : 1}

    n_patterns = len(data_X)
    print("Number of data entries", n_patterns)
    #Arranging the data in a form that can be provided to the Deep Learning architecture using Keras.
    X = np.reshape(data_X, (n_patterns, seq_length, 1))
    Y = data_Y

    print("Data prepared")

    #Creating the Keras model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(256))
    model.add(Dropout(0.7))
    model.add(Dense(Y.shape[1], activation='softmax'))

    filename = folder+"/Locality.hdf5"
    model.load_weights(filename)
    #sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    #sgd = optimizers.SGD(lr=0.001, decay=0.0, momentum=0.0, nesterov=False)

    #filename = "mt-weights-improvement=10-0.023081.hdf5"
    #model.load_weights(filename)
    #model.compile(loss='mean_squared_error', optimizer=sgd)

    print("Model created.....")
    print("Testing.....")

    if(out_nodes==17):
        checking = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
                '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
                '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
                '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1}#,
               # '4001-5000' : 1, '5001-6000' : 1}
    else:
        checking = {'0-25': 1, '26-50': 1, '51-75': 1, '76-100': 1, '101-150': 1,
                '151-200' : 1, '201-300' : 1, '301-400' : 1, '401-500' : 1,
                '501-600' : 1, '601-700' : 1, '701-800' : 1, '801-900' : 1,
                '901-1000' : 1, '1001-2000' : 1, '2001-3000' : 1, '3001-4000' : 1,
                '4001-5000' : 1, '5001-6000' : 1}

    # #For testing
    # checking = {0:'0-25', 1:'26-50', 2:'51-75', 3:'76-100', 4:'101-150', 5:'151-200',
    #             6:'201-300', 7:'301-400', 8:'401-500', 9:'501-600', 10:'601-700',
    #             11:'701-800', 12:'801-900', 13:'901-1000', 14:'1001-2000', 15:'2001-3000',
    #             16:'3001-4000'}#, 17:'4001-5000', 18:'5001-6000'}

    accurate = 0
    '''
    prediction = model.predict(X, verbose=0)
    prediction = np.array(prediction)
    index = prediction.argmax(-1)
    true_labels = Y.argmax(-1)

    accuracy = np.mean(index == true_labels)
    print "Accuracy : ", accuracy*100, "%"
    '''

    n_patterns=30

    #Testing 1 by 1.
    for i in range(n_patterns):
        pattern = data_X[i]
        if(i==0):
            for j in pattern:
                pred_val_temp.append(j*max)
                lower_bound.append(j*max)
                upper_bound.append(j*max)
                real_val_temp.append(j)
        else:
            real_val_temp.append(pattern[5])
        # for j in pattern:
        #     if j not in real_val_temp:
        #         real_val_temp.append(j)

        test_X = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(test_X, verbose=0)[0]
        index = np.argmax(prediction)
        pred_val_temp.append(index)
        if (i > 499) and (i % 500 == 0):
            print(i, " data points tested.")
        #print checking[index], np.argmax(data_Y[i])
        #print index, np.argmax(data_Y[i])

        if index == np.argmax(data_Y[i]):
            accurate += 1

    print("Accuracy : ", (accurate/float(n_patterns))*100)
    print("LENGTH OF ",len(real_val_temp))
    for j in real_val_temp:
        real_val.append(j*max)
    for j in pred_val_temp[6:]:
        # print(j)
        lb=support[j][0]
        ub=support[j][1]
        lower_bound.append(lb)
        upper_bound.append(ub)

    return real_val,lower_bound,upper_bound

myFolder={'gps1':'locality1','gps2':'locality2','gps3':'locality3'}
myOutput={'gps1':19,'gps2':17,'gps3':19}
myMax={'gps1':5853,'gps2':3638,'gps3':5928}


def plotting(real,upper,lower):
    for i in range(len(real)):
        if (lower[i] > upper[i]) or (lower[i] < real[i]):
            upper[i]+=500
    plt.plot(lower, 'r--')
    plt.plot(upper, 'bs')
    plt.plot(real, 'bs')
    plt.xlabel('Half hour')
    plt.ylabel('Power consumption in KWh')
    store = []
    anomalies = 0
    for i in range(len(real)):
        if (lower[i] > upper[i]) or (lower[i] < real[i]):
            anomalies += 1
            tup = (i+1, lower[i])
            store.append(tup)
    # for i in range(anomalies):
    #     plt.annotate('Anomaly', xy=(store[i][0], store[i][1]), xytext=(store[i][0]+1, store[i][1]+1), 
    #     arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.savefig("Graph1.png")
    plt.show()


def getGPS(x,t):
    # y=x.split(',')
    # for j in range(len(y)):
    #   y[j]=float(y[j])
    # folder=myFolder[x]
    real_val,lower_bound,upper_bound=calc(myFolder[x],myOutput[x],myMax[x],t)
    print(real_val)
    print(lower_bound)
    plotting(real_val,upper_bound,lower_bound)
    # plotting(upper_bound,real_val,lower_bound)
    # print("REAL VALUES")
    # print((real_val))
    # print("LOWER BOUND")
    # print((lower_bound))
    # print("UPPER BOUND")
    # print((upper_bound))
    

# getGPS('gps1',68)
# getGPS('gps1',12)
# getGPS('gps1',8)