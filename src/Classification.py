# Classification.py

import os, sys
from os.path import join, realpath, dirname, isdir
import csv
import numpy as np
from doc_globals import*

#loads the features from the csv file created during extraction
#
#@result:   list of lists of features, list of corresponding classes and filenames
def load_data(features_file):
    
    with open(features_file, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)
      
    num_features = len(data[0])-2
            
    features = [item[:num_features] for item in data]

    features = generate_error_features(features)     
    
    classes = [item[num_features] for item in data]
    filenames = [item[num_features+1] for item in data]
    
    return features, classes, filenames
    
#generates the error features and replaces the nan values
#
#@result:   list of features with the error features included
def generate_error_features(features):
    
    error_feature = [0.0] * len(features)
    
    for i, x in enumerate(features):
        for j, e in enumerate(x):
            if e == 'nan':
                error_feature[i] = 1.0
                x[j] = 1.0                
    
    features = [x + [error_feature[i]] for i, x in enumerate(features)]
    return features

def get_feature_vals_and_classes(filedir):
	features, classes, files = load_data(features_file) 
	features = [[float(j) for j in i] for i in features]
	classes = [float(i) for i in classes]
	len_feat = len(features[0])

	for i in range(0, len_feat):
		max_nor=max(map(lambda x: x[i], features))
		if max_nor > 1:
			min_nor=min(map(lambda x: x[i], features))
			for f in features: (f[i] - min_nor)/(max_nor-min_nor)

	features=np.array([np.array(xi) for xi in features])
	return features, classes, files

#this function will initialize the neural network
# @parram input_dim:    number of modules for the input vector
def getNN(input_dim, hidden_dims):
    network = NN()
    network.initializeNN(input_dim, hidden_dims)
    return network

def plot(nodes, epochs, bias, features, classes, ):
    
    import matplotlib.pyplot as plt
    import time

    fig = 0
        
    if nodes:
        num_nodes = [10, 20, 50, 100, 200, 500]
        ll = len(num_nodes)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
        
        for i, x in enumerate(num_nodes):
            network = getNN(len(features[0]), x)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[1][3] = network.trainNN(features, np.array(classes), 100, .5)
            tm[i] = time.time() - start_time
            
        fig+=1
            
        plt.figure(fig)
        plt.plot(num_nodes, ac)
        plt.ylabel('accuracy (%)')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(ac)-1, max(ac)+1 ])
        plt.savefig(join(RESULT_PATH,'nodesAc.jpg'))
        
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_nodes, f1)
        plt.ylabel('f1')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(f1)-.1, max(f1)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'nodejoin(RESULT_PATH,sF1.jpg'))
        
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_nodes, pr)
        plt.ylabel('precision')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(pr)-.1, max(pr)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'nodesPr.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_nodes, rc)
        plt.ylabel('recall')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+50, min(rc)-.1, max(rc)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'nodesRc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_nodes, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('number of examples')
        plt.xlabel('number of nodes')
        #plt.axis([ 0, num_nodes[ll-1]+50, 0, 300 ])
        
        plt.savefig(join(RESULT_PATH,'nodesEx.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_nodes, tm)
        plt.ylabel('time in seconds')
        plt.xlabel('number of nodes')
        plt.axis([ 0, num_nodes[ll-1]+100, min(tm), max(tm) ])
        
        plt.savefig(join(RESULT_PATH,'nodesTm.jpg'))
                
    if epochs:
        num_epochs = [50, 100, 200, 500, 1000, 2000]
        ll = len(num_epochs)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll

        for i, x in enumerate(num_epochs):
            network = getNN(len(features[0]), 100)
            start_time = time.time()
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[i][3] = network.trainNN(features, np.array(classes), x, .5)
            tm[i] = time.time() - start_time
            
        fig+=1
        
        plt.figure(fig)
        plt.plot(num_epochs, ac)
        plt.ylabel('accuracy (%)')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(ac)-1, max(ac)+1 ])
        
        plt.savefig(join(RESULT_PATH,'epochsAc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, f1)
        plt.ylabel('f1')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(f1)-.1, max(f1)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'epochsF1.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, pr)
        plt.ylabel('precision')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(pr)-.1, max(pr)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'epochsPr.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, rc)
        plt.ylabel('recall')
        plt.xlabel('number of epochs (times 10)')
        plt.axis([ 0, num_epochs[ll-1]+500, min(rc)-.1, max(rc)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'epochsRc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(num_epochs, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('number of examples')
        plt.xlabel('number of epochs (times 10)')
        #plt.axis([ 0, num_epochs[ll-1]+500, 0, 300 ])
        
        plt.savefig(join(RESULT_PATH,'epochsEx.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(num_epochs, tm)
        plt.ylabel('time in seconds')
        plt.xlabel('number of epochs')
        plt.axis([ 0, num_epochs[ll-1]+500, min(tm), max(tm) ])
        
        plt.savefig(join(RESULT_PATH,'epochsTm.jpg'))
            
    if bias:
        bias_cut = [.2, .3, .4, .5]
        ll = len(bias_cut)
        ac = [None]*ll
        f1 = [None]*ll
        pr = [None]*ll
        rc = [None]*ll
        ex = np.empty([ll,4])
        tm = [None]*ll
                
        for i, x in enumerate(bias_cut):
            network = getNN(len(features[0]), 100)
            ac[i], f1[i], pr[i], rc[i], ex[i][0], ex[i][1], ex[i][2], ex[i][3] = network.trainNN(features, np.array(classes), 100, x)

        fig+=1

        plt.figure(fig)
        plt.plot(bias_cut, ac)
        plt.ylabel('accuracy (%)')
        plt.xlabel('bias cut')
        plt.axis([ bias_cut[0]-.01, bias_cut[ll-1]+.01, min(ac)-1, max(ac)+1 ])
        
        plt.savefig(join(RESULT_PATH,'biasAc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(bias_cut, f1)
        plt.ylabel('f1')
        plt.xlabel('bias cut')
        plt.axis([ bias_cut[0]-.01, bias_cut[ll-1]+.01, min(f1)-.1, max(f1)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'biasF1.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        plt.plot(bias_cut, pr)
        plt.ylabel('precision')
        plt.xlabel('bias cut')
        plt.axis([ bias_cut[0]-.01, bias_cut[ll-1]+.01, min(pr)-.1, max(pr)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'biasPr.jpg'))

        fig+=1
        
        plt.figure(fig)
        plt.plot(bias_cut, rc)
        plt.ylabel('recall')
        plt.xlabel('bias cut')
        plt.axis([ bias_cut[0]-.01, bias_cut[ll-1]+.01, min(rc)-.1, max(rc)+.1 ])
        
        plt.savefig(join(RESULT_PATH,'biasRc.jpg'))
        
        fig+=1        
        
        plt.figure(fig)
        lineObjects = plt.plot(bias_cut, ex)
        plt.legend(lineObjects, ('tn', 'tp', 'fn', 'fp'))
        plt.ylabel('number of examples')
        plt.xlabel('bias cut')
        #plt.axis([ 0, bias_cut[0]+.5, 0, 300 ])
        
        plt.savefig(join(RESULT_PATH,'biasEx.jpg'))
        
    plt.show()


if __name__ == "__main__":
	args = sys.argv
	len_args = len(args)

	usage = "python Classification.py <feature_file.csv>"

	if(not(len_args==2)):
		print(usage)
		sys.exit(1)

	features_file = args[1]
	if not os.path.isfile(features_file):
		print("Error: Features file doesn't exist.")
		exit();

	f_vals, f_classes, files = get_feature_vals_and_classes(features_file)

	from simple_neural_network import NN
	hidden_dims = 500
	num_epochs = 500
	conf_thresh = 0.5

	print("Initiating Neural Network")
	network = getNN(input_dims=len(f_vals[0]), hidden_dims=hidden_dims)
	print("Starting training.")
	network.trainNN(f_vals,np.array(f_classes), files, num_epochs, conf_thresh)
	print("Training done!")

	# # Do some longer analysis
	# plot(nodes=True, epochs=True, bias=True, features, classes)

	# from log_reg import Log_Reg
	
	# print("Initiating Logistic Regression")
	# lg = Log_Reg()
	# print("Starting training.")
	# lg.kfold_log_reg(f_vals,np.array(f_classes), files)
	# print("Starting Evaluation!")


