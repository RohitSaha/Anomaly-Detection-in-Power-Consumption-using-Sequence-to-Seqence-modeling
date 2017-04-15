import numpy as np

with np.load("BackupdataOneHousehold.npz") as file:
    d_X = file["backup_X"]
    d_Y = file["backup_Y"]

with np.load("SecSetLocality.npz") as file:
    d_X2 = file["backup_X"]
    d_Y2 = file["backup_Y"]

d_Y = np.pad(d_Y,((0,0),(0,1)),mode='constant', constant_values=0)

print(d_X,d_X.shape)
print(d_Y,d_Y.shape)

d_X = np.vstack((d_X,d_X2))
d_Y = np.vstack((d_Y,d_Y2))

print("Vstacked both localities")
print(d_X,d_X.shape)
print(d_Y,d_Y.shape)

# Shuffle
random_indexes = list(range(len(d_X)))
np.random.shuffle(random_indexes)
data_X = np.ndarray(shape=(len(d_X), 6))
data_Y = np.ndarray(shape=(len(d_X), 8))
for i, j in zip(random_indexes, range(len(data_X))):
    data_X[j] = d_X[i]
    data_Y[j] = d_Y[i]

# Normalize
max_val = 661
# data_X = data_X/661.0
# data_X = data_X/597.5

print("\n Split data into classes")
data_X1, data_Y1 = [], []
data_X2, data_Y2 = [], []
data_X3, data_Y3 = [], []
data_X4, data_Y4 = [], []
data_X5, data_Y5 = [], []
data_X6, data_Y6 = [], []
for i in range(len(data_Y)):
    if (data_Y[i][0] == 1):
        data_X1.append(data_X[i])
        data_Y1.append(data_Y[i])

    elif (data_Y[i][1] == 1):
        data_X2.append(data_X[i])
        data_Y2.append(data_Y[i])

    elif (data_Y[i][2] == 1):
        data_X3.append(data_X[i])
        data_Y3.append(data_Y[i])

    elif (data_Y[i][3] == 1):
        data_X4.append(data_X[i])
        data_Y4.append(data_Y[i])

    elif (data_Y[i][4] == 1):
        data_X5.append(data_X[i])
        data_Y5.append(data_Y[i])

    else:
        data_X6.append(data_X[i])
        data_Y6.append(data_Y[i])

data_X1, data_Y1 = np.array(data_X1), np.array(data_Y1)
data_X2, data_Y2 = np.array(data_X2), np.array(data_Y2)
data_X3, data_Y3 = np.array(data_X3), np.array(data_Y3)
data_X4, data_Y4 = np.array(data_X4), np.array(data_Y4)
data_X5, data_Y5 = np.array(data_X5), np.array(data_Y5)
data_X6, data_Y6 = np.array(data_X6), np.array(data_Y6)

split = int(0.8*len(data_X1))
train_X = np.vstack((data_X1[:split],data_X2[:split],data_X3[:split],data_X4[:split],data_X5[:split]))
train_Y = np.vstack((data_Y1[:split],data_Y2[:split],data_Y3[:split],data_Y4[:split],data_Y5[:split]))

test_X = np.vstack((data_X1[split:],data_X2[split:],data_X3[split:],data_X4[split:],data_X5[split:]))
test_Y = np.vstack((data_Y1[split:],data_Y2[split:],data_Y3[split:],data_Y4[split:],data_Y5[split:]))

np.savez("Split_TwoLocalities.npz", train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

print(train_X,train_X.shape)
print(train_Y,train_Y.shape)
print(test_X,test_X.shape)
print(test_Y,test_Y.shape)