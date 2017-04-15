import pandas as pd
import numpy as np

dataset = pd.read_csv("electricityconsumptionbenchmarkssurveydataaergovhack.csv").as_matrix()

locality = (set(dataset[:,0]))

# locality = [5636,4192,8927,6520,1098]
# locality = [8927]

# choose locality houses from dataset
print("Choosing House Localities")
data = []
for row in dataset:
    for l in locality:
        if row[0] == l:
            data.append(row)

data = np.array(data)
data = data[:,3:]
print("rows ",len(data))

# BoxPlot
unroll = np.reshape(data,(data.shape[0]*data.shape[1]))
unroll = np.sort(unroll)
print("\n Boxplot")
l1 = int(len(unroll)/2)
m1 = unroll[l1]
print("median ",m1)
d1,d2 = unroll[:l1],unroll[l1:]
l2 = int(len(d1)/2)
m2 = unroll[l2]
print("lower quartile ",m2)
l3 = int(len(d2)/2)
m3 = unroll[l1+l3]
print("upper quartile ",m3)
iqr = m3-m2
outlierLim = (m3+1.5*iqr)
print("maximum ",outlierLim)
outlierLimMin = m2-1.5*iqr
print("minimum ",outlierLimMin)

# Choose rows removing excess of 0s and outliers
print("\n Removed Noise")
data2=[]

for row in range(len(data)):
    ctr = 0
    for val in range(len(data[row])):
        if data[row][val]==0:
            ctr+=1
        # if data[row][val]>=outlierLim:
        #     data[row][val] = outlierLim
    if ctr==0:
        data2.append(data[row])
data2 = np.array(data2)
print("rows",len(data2))
data = data2
del data2


# Choosing Intervals
print("\n Choosing Intervals")
def no_classes():
    unroll = np.reshape(data, (data.shape[0] * data.shape[1]))
    sortUnroll = np.sort(unroll)
    num_classes = 6
    perClass = int(len(sortUnroll)/num_classes)
    c = []
    for i in range(0,len(sortUnroll),perClass):
        c.append(sortUnroll[i:i+perClass])
    intervals = []
    for i in c:
        print(len(i))
    for i in c:
        intervals.append(max(i))
    print (intervals)
no_classes()
# intervals = ['0-70', '71-116', '117-165', '166-253', '254-427', '428-661']
# intervals = ['0-38', '39-63', '64-104', '105-191', '192-477', '478-598']
# intervals = ['0-25', '26-50', '51-75', '76-100', '101-150',
#         '151-200', '201-300' , '301-400' , '401-500',
#         '501-600' , '601-700' ]

intervals = ['0-100', '101-200 ', '201-300', '301-400', '401-500', '501-600', '601-700','701-800']

print(intervals)

# i/p-o/p pairs
seq_length = 6
unroll = np.reshape(data, (data.shape[0] * data.shape[1]))
new_unroll=[]
for i in unroll:
    if not i>=outlierLim:
        new_unroll.append(i)
new_unroll = np.array(new_unroll)
unroll = new_unroll
del new_unroll

def calculate_1_hot(value):
    hot_1_vector = []
    for i in intervals:
        values = i.split('-')
        if value >= int(values[0]) and value <= int(values[1]):
            hot_1_vector.append(1)
        else:
            hot_1_vector.append(0)
    return hot_1_vector

print("------------- class of 106",calculate_1_hot(106))

data_X = np.zeros((1, 6), 'float')
data_Y = np.zeros((1, len(intervals)), 'float')
for i in range(0, len(unroll) - seq_length, 1):
    if i%10000 == 0:
        print(i,"/",len(unroll))
    data_X = np.vstack((data_X, unroll[i: i+seq_length]))
    av = np.mean(unroll[i: i+seq_length])
    get_1_hot_vector = calculate_1_hot(av)
    data_Y = np.vstack((data_Y, get_1_hot_vector))

data_X = data_X[1:, :]
data_Y = data_Y[1:, :]

print(data_X, len(data_X))
print(data_Y, len(data_Y))

np.savez("AllLocality.npz", backup_X=data_X, backup_Y=data_Y)