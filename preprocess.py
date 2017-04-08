import numpy as np
import pandas as pd
import pickle

f = "electricityconsumptionbenchmarkssurveydataaergovhack.csv"

data = pd.read_csv(f).values

# Delete rows with 0's
newD = []
for i in data:
    if (i[3:] == 0).all():
        pass
    else:
        newD.append(i)
newD = np.array(newD)

# Box Plot Extremes
d = newD[:,3:]
d = np.reshape(d,(d.shape[0]*d.shape[1]))
d = np.sort(d)
l1 = int(len(d)/2)
m1 = d[l1]
d1,d2 = d[:l1],d[l1:]
l2 = int(len(d1)/2)
m2 = d[l2]
l3 = int(len(d2)/2)
m3 = d[l1+l3]
outlierLim = (m3*1.5)

# For LSTM - remove 0s and outliers
d2 = []
D = newD[:,3:]
D = np.reshape(D,(D.shape[0]*D.shape[1]))
print (len(D))
for i in range(len(D)):
    if D[i]!=0 and D[i]<outlierLim:
        d2.append(D[i])
d2 = np.array(d2)
# print(d2)

g = open("oneArrayPowers",'wb')
pickle.dump(d2,g)
g.close()

# g = open("oneArrayPowers",'rb')
# d2 = pickle.load(g)
# print(d2)


# print(len(d2))
# print(len(d2)/len(D))

# Pre-processed dataset after removing outliers
# data2 = []
# for i in range(len(data)):
#     ctr = 0
#     for j in range(3,len(data[i]),1):
#         if data[i][j] > outlierLim:
#             ctr+=1
#     if ctr==0:
#         data2.append(data[i])
#
# data2 = np.array(data2)
# print(len(data2)) 6735
# print(len(data))  22952


