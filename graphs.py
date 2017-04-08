import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = "electricityconsumptionbenchmarkssurveydataaergovhack.csv"
data = pd.read_csv(f).values
households = data[:,0]

hrs = [0 for i in range(48)]

data = data[:,3:]

for i in range(48):
    for j in range(len(data)):
        hrs[i] += data[j][i]

hrs = np.array(hrs)
hrs = hrs/len(data)

x_axis = [i for i in range(48)]

# a = ":00"
# b=":30"
# x_ticks = []
# i=0
# while i<24:
#     x_ticks.append(str(i)+a)
#     x_ticks.append("     ")
#     i+=1

# plt.xticks(x_axis,x_ticks)
plt.plot(x_axis,hrs,'ro')
plt.xlabel("Half-Hours")
plt.ylabel("Average Power Consumption")
plt.savefig("graph.png")