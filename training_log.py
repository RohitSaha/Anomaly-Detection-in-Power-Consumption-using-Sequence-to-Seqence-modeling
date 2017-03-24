import glob
import pickle

#Program to extract information about training from the weight files stored in the folders.
#Information format : Epoch-Loss

info = {}
weight_files = glob.glob('More_Dense_Net/*.hdf5')

for files in weight_files:
    s = (files[:-5].split('='))[1].split('-')
    info[int(s[0])] = float(s[1])

with open("DenseNet_training_log.pkl", 'wb') as file:
    pickle.dump(info, file)





