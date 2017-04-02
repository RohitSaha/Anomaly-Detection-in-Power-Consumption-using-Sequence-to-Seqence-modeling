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

###################################   getting the data ##############################################################
##Type Cast into float
for i in range(1,len(train_input)):
	for j in range(3,len(train_input[i])):
		train_input[i][j]=float(train_input[i][j])
for i in range(1,len(train_input)):
	for j in range(0,1):
		train_input[i][j]=float(train_input[i][j])
train_input=train_input[1:,3:]
train_output=KMeans(n_clusters=2,random_state=0).fit(train_input)
train_output=train_output.labels_
ti=[]
print(type(train_input))
input()
# train_input=train_input.transpose()
# print(len(train_input[0]))
# input()
# def t2():
# 	trained_in=[]
# 	for i in range(len(train_input[0])):
for i in train_input:
	temp=[]
	for j in i:
		temp.append([j])
	ti.append(np.array(temp))

train_input=ti

zero=0
one=0
for i in train_output:
	if(i==0):
		zero+=1
	else:
		one+=1
print("zero is",zero,"    one is",one)
to=[]
for i in train_output:
	temp=[0]*2
	if(i==0):
		temp[0]=1
	else:
		temp[1]=1
	to.append(temp)
train_output=to
zero=0
one=0
for i in train_output:
	if(i[0]==1):
		zero+=1
	else:
		one+=1
print("zero is",zero,"    one is",one)


del to

NUM_EX=20000
num_hidden=24
output_dim=2
alpha=0.05
nb_epoch=10000
batch_size=50
# test_input=train_input[NUM_EX:]
# train_input=train_input[:NUM_EX]
# test_output=train_output[NUM_EX:]
# train_output=train_output[:NUM_EX]

def Rohit_Slice(tr_in,tr_ou):
	zero_in=[]
	zero_out=[]
	one_in=[]
	one_out=[]
	for i in range(len(tr_in)):
		if(tr_ou[i][0]==1):
			zero_in.append(tr_in[i])
			zero_out.append(tr_ou[i])
		else:
			one_in.append(tr_in[i])
			one_out.append(tr_ou[i])

	train_input=[]
	train_output=[]
	test_input=np.vstack((zero_in[:400],one_in[400:]))
	length=len(one_in)
	test_output=np.vstack((zero_out[400:length],one_out[400:]))
	print(len(one_in))
	return(tr_in,tr_ou,test_input,test_output)

# train_input,train_output,test_input,test_output=Rohit_Slice(train_input,train_output)
# input()

# pickle.dump([train_input,train_output,test_input,test_output],file)
# train_input,train_output,test_input,test_output=pickle.load(file)


print("DATA Ready")

# #############################################################
print("MAKING COMPUTATION GRAPH")

###################       making ops

################           placeholder
with tf.name_scope("Inputs"):
	x=tf.placeholder(tf.float32,[None,48,1],name='input')
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
	y_=tf.nn.softmax(tf.add(tf.matmul(last,weight),bias),name='prediction')

##loss
with tf.name_scope('errors'):
	cross_entropy=-tf.reduce_sum(y*tf.log(y_))
with tf.name_scope('TRAINING'):
	train_step=tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cross_entropy)

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
	# if(i%10==0):
		##calculating the accuracy over test set
		acc=sess.run(accuracy,feed_dict={x:test_input[ptr_acc:ptr_acc+2],y:test_output[ptr_acc:ptr_acc+2]})
		# test_writer.add_summary(m,i)
		print("Accuracy= ",acc,"epoch",i)#,"prediction",lol)
		ptr_acc+=2
		if(ptr_acc>=len(test_input)):
			ptr_acc=0
		# print(i)

# # print(var.get_name for var in my_var)	
