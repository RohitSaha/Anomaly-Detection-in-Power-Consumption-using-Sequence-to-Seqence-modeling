# import tensorflow as tf
import numpy as np
# from random import shuffle
# import random
import pickle
import pandas as pd
# from sklearn.cluster import KMeans
# import csv




all_id=[]


sum_days=[]
train_input=pd.read_csv("electricityconsumptionbenchmarkssurveydataaergovhack.csv",sep=',',header=None).as_matrix()
new=[]
for i in range(len(train_input)):
	if(train_input[i][0]=='6520' or train_input[i][0]=='1098'):# or train_input[i][0]=='2447' or train_input[i][0]=='10746'):
		new.append(train_input[i])
train_input=np.array(new)
train_input=train_input[:,3:]
print(len(train_input[0]))
##it is for one house
print("reached")
# input()	
print(train_input.shape)
train_input=np.reshape(train_input,train_input.shape[0]*train_input.shape[1])
print(train_input.shape)
with open('objs_4.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(train_input, f)