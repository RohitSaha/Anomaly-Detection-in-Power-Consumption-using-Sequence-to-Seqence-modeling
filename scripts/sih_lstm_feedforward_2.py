import tensorflow as tf
import numpy as np
from random import shuffle
import random
import pickle
import pandas as pd
from sklearn.cluster import KMeans
import csv




all_id=[]
sum_days=[]
train_input=pd.read_csv("electricityconsumptionbenchmarkssurveydataaergovhack.csv",sep=',',header=None).as_matrix()
new=[]
for i in range(len(train_input)):
	if(train_input[i][0]=='8927'):
		new.append(train_input[i])
train_input=np.array(new)
train_input=train_input[:,3:]
print(len(train_input[0]))
##it is for one house
print("reached")
input()	
print(train_input.shape)
train_input=np.reshape(train_input,train_input.shape[0]*train_input.shape[1])
print(train_input.shape)
input()

for i in range(len(train_input)):
	train_input[i]=float(train_input[i])

###################################   getting the data ##############################################################
print("yo")
input()
def create_one_hot(val):
	out=[0]*6
	if(val<1000):
		out[0]=1
	elif(val>1000 and val<2000):
		out[1]=1
	elif(val>2000 and val<3000):
		out[2]=1
	elif(val>3000 and val<4000):
		out[3]=1
	elif(val>4000 and val<5000):
		out[4]=1
	elif(val>5000 and val<6000):
		out[5]=1
	return out
def create_data():
	trained_in=[]
	trained_out=[]
	for i in range(0,len(train_input)-6):
		temp=[]
		temp.append([train_input[i]])
		temp.append([train_input[i+1]])
		temp.append([train_input[i+2]])
		temp.append([train_input[i+3]])
		temp.append([train_input[i+4]])
		temp.append([train_input[i+5]])
		trained_out.append([train_input[i+6]])
		# get_one_hot=create_one_hot(train_input[i+6])
		# trained_out.append(get_one_hot)
		trained_in.append(temp)
	return trained_in,trained_out
print("shakalaka boom boom")
x,y=create_data()
train_input=x
train_output=y
print(len(train_input))
print(len(train_output))
input()


NUM_EX=24524
num_hidden=10
output_dim=1
alpha=0.001
nb_epoch=10000
batch_size=50
test_input=train_input[NUM_EX:]
train_input=train_input[:NUM_EX]
test_output=train_output[NUM_EX:]
train_output=train_output[:NUM_EX]



print("DATA Ready")

# #############################################################
print("MAKING COMPUTATION GRAPH")

###################       making ops

################           placeholder
with tf.name_scope("Inputs"):
	x=tf.placeholder(tf.float32,[None,6,1],name='input')
	y=tf.placeholder(tf.float32,[None,output_dim],name='targets')

##cell state
with tf.name_scope("RNNS"):
	cell=tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
	val,state=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
	##val will be of the order: batch,sequence,output@sequence

	##preprocessing the val
	val=tf.transpose(val,[1,0,2])
	print(val.get_shape())
	##removing the sequence wala part
	last=tf.gather(val,int(val.get_shape()[0]-1),name='hiddenstatesforbatches')
	print(last.get_shape())

with tf.name_scope("feed_forward"):
	##creating weights and biases for the feedforward net
	weight=tf.Variable(tf.random_normal([num_hidden,output_dim]),name='W')
	bias=tf.Variable(tf.random_normal([output_dim]),name='bias')
	tf.summary.histogram('weights',weight)
	tf.summary.histogram('bias',bias)

	##final prediction
	y_=tf.nn.relu(tf.add(tf.matmul(last,weight),bias),name='prediction')

##loss
with tf.name_scope('errors'):
	cross_entropy=tf.reduce_mean(tf.squared_difference(y_,y))
	# cross_entropy=-tf.reduce_sum(y*tf.log(y_))
with tf.name_scope('TRAINING'):
	train_step=tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(cross_entropy)

##calculating accuracy
with tf.name_scope('accuracy'):
	correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

# tf.summary.scalar('cost',cross_entropy)
# tf.summary.scalar('accuracy',accuracy)
# with tf.variable_scope("RNN/LSTMCell") as vs:
# 	my_var=[v for v in tf.trainable_variables()
# 			if v.name.startswith(vs.name)]
# for i in my_var:
# 	tf.summary.histogram(str(i.name),i)

# merged_summary=tf.summary.merge_all()

# train_writer=tf.summary.FileWriter('/tmp/rnn_paper/count/train')
# test_writer=tf.summary.FileWriter('/tmp/rnn_paper/count/test')
init_op=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init_op)
# train_writer.add_graph(sess.graph)
# test_writer.add_graph(sess.graph)

# # x=tf.get_collection(tf.	GraphKeys.VARIABLES,scope='RNNS')
# # my_var=tf.trainable_variables()

ptr=0
ptr_acc=0
for i in range(nb_epoch):
	if(ptr+batch_size<=len(train_input)):
		batch_x=train_input[ptr:ptr+batch_size]
		batch_y=train_output[ptr:ptr+batch_size]
		ptr+=batch_size
		t=sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
		# train_writer.add_summary(m,i)
	if(i%10==0):
		##calculating the accuracy over test set
		cst,acc=sess.run([cross_entropy,accuracy],feed_dict={x:test_input[ptr_acc:ptr_acc+2],y:test_output[ptr_acc:ptr_acc+2]})
		# test_writer.add_summary(m,i)
		print("Accuracy= ",acc,"epoch",i,"cst",cst)#,"prediction",lol)
		ptr_acc+=2
		if(ptr_acc>=len(test_input)):
			ptr_acc=0
		# print(i)
c1=np.array([[426.0],[396.0],[340.0],[392.0],[348.0],[378.0]])
c1=np.reshape(c1,(1,6,1))
c2=np.array([[362.0]])
chk=sess.run(y_,feed_dict={x:c1,y:c2})
print(chk)
# # print(var.get_name for var in my_var)	
