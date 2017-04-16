from firebase import firebase
import random
##list of names of house
# names=['Rohit','Raghav','Rohan','Soham','Fenil','Flash','Batman','Aquaman']
gps=[]
# x="abcd,kl"
# y=x.split(',')
# print()
##
##fetching the data from firebase to python
firebase=firebase.FirebaseApplication('https://sihuser-11acb.firebaseio.com')
result=firebase.get('/complaints',None)
# result=firebase.get('GPS',None)
for i in result.keys():
	x=result[i]['GPS']
	y=x.split(',')
	for j in range(len(y)):
		y[j]=float(y[j])

	gps.append(y)

print(gps)

