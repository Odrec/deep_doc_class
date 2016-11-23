# Classification.py

import os, sys
from os.path import join, realpath, dirname, isdir, basename
if(isdir('/usr/lib/python3.5/lib-dynload')):
    sys.path.append('/usr/lib/python3.5/lib-dynload')
import csv
import numpy as np
import pandas as pd
from doc_globals import* 

from tsne import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties
import colorsys

from prettytable import PrettyTable
#loads the features from the csv file created during extraction
#
#@result:   list of lists of features, list of corresponding classes and filenames
def load_data(features_file):
    features=pd.read_csv(features_file, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    f_names = list(features)
    np_features = features.as_matrix(f_names[:-2])
    np_classes = np.array(features[f_names[-2]].tolist())
    filenames = features[f_names[-1]].tolist()

    # error_col, error_names = generate_error_features(np_features)
    # np_features = np.append(np_features, error_col, axis=1)
    # f_names = f_names.extend(error_names)

    return np_features, np_classes, filenames, f_names[:-2]
    
#generates the error features and replaces the nan values
#
#@result:   list of features with the error features included
def generate_error_features(features):
    
    error_feature = np.zeros((len(features),1))
    
    for i in range(0,len(features)):
        if(np.nan in features[i]):
            error_feature[i] = 1.0
    
    return error_feature, ["error_mod"]

def replace_nan_mean(features):
    col_mean = np.nanmean(features,axis=0)
    inds = np.where(np.isnan(features))
    features[inds]=np.take(col_mean,inds[1])
    return features

def replace_nan_weighted_dist(features):
    #TODO:
    return replace_nan_mean(features)

def norm_features(features):
    len_feat = len(features[0])
    max_nor=np.amax(features, axis=0)
    min_nor=np.amin(features, axis=0)
    for i in range(0, len_feat):
        f_range = (max_nor[i]-min_nor[i])
        if(f_range>0):
            features[:,i] = (features[:,i]-min_nor[i])/f_range
        else:
            print(i)
    return features

#this function will initialize the neural network
# @parram input_dim:    number of modules for the input vector
def getNN(input_dim, hidden_dims):
    network = NN()
    network.initializeNN(input_dim, hidden_dims)
    return network

def pca(data, dims=3):
    (n, d) = data.shape;
    data = data - np.tile(np.mean(data, 0), (n, 1));
    cov = np.dot(data.T, data)/(n-1)
    # create a list of pair (eigenvalue, eigenvector) tuples
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_pairs = []
    for x in range(0,len(eig_val)):
        eig_pairs.insert(x, (np.abs(eig_val[x]),  np.real(eig_vec[:,x])))
    
    # sort the list starting with the highest eigenvalue
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)          
    M = np.hstack((eig_pairs[i][1].reshape(d,1) for i in range(0,dims)))

    data_trans = np.dot(data, M)
    return data_trans, eig_pairs

def tsne(data, dims=3, initial_dims=50, max_iter=300, perplexity=30.0):
    tsne = TSNE(data, no_dims=dims, initial_dims=initial_dims, max_iter=max_iter, perplexity=perplexity)
    data_trans = tsne.run()
    return data_trans

def scatter_3d(data, classes, filepath):
    (n,d) = np.shape(data)
    assert d == 3 , 'Need to have 3 dimensions'
    # create a plot for the average over each cycle type
    fig = plt.figure('scatter3D')
    dpi = fig.get_dpi()
    fig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))
    ax = fig.add_subplot(111, projection='3d')
    
    maxTicks = 5
    loc = plt.MaxNLocator(maxTicks)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(loc)
    
    ax.set_title('scatter3D')
    # add labels
    ax.set_xlabel('\n' +'Dim1', linespacing=1.5)
    ax.set_ylabel('\n' +'Dim2', linespacing=1.5)
    ax.set_zlabel('\n' +'Dim3', linespacing=1.5)
    
    colors = get_colors(2)
    t_data = data[:,:][classes==1]
    n_data = data[:,:][classes==0]
    ax.scatter(t_data[:,0],t_data[:,1],t_data[:,2],s=10, color = colors[0], label = "protected", edgecolor=colors[0])
    ax.scatter(n_data[:,0],n_data[:,1],n_data[:,2],s=10, color = colors[1], label = "not_protected", edgecolor=colors[1])
         
    # add a legend and title
    ax.view_init(elev=10, azim=-90)
    
    fontP = FontProperties()
    fontP.set_size('medium')
    
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0,0.7), prop=fontP)
    ax.grid('on')
    
    fig.savefig(filepath+"_e10a90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=80, azim=-90)
    fig.savefig(filepath+"_e80a-90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=15, azim=-15)
    fig.savefig(filepath+"_e15a-15", bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()

def create_boxplot(data, collumn_names, filepath):

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)

    ## add patch_artist=True option to ax.boxplot() 
    ## to get fill color
    bp = ax.boxplot(data, whis=[10,90], vert=False, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='.', color='#e7298a', alpha=0.2)

    ## Custom x-axis labels
    ax.set_yticklabels(collumn_names)
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Save the figure
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

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

    folder = basename(features_file).split(".")[0]
    folder = join(RESULT_PATH, folder)
    if(not(isdir(folder))):
        os.mkdir(folder)

    f_vals, f_classes, files, f_names = load_data(features_file)

    f_vals = replace_nan_mean(f_vals)
    f_vals = norm_features(f_vals)


    red_vals = f_vals[:,[0,2,3,4,5,6,7,9,14,16]]
    red_names = [f_names[idx] for idx in [0,2,3,4,5,6,7,9,14,16]]

    from simple_neural_network import NN
    hidden_dims = 500
    num_epochs = 500
    conf_thresh = 0.5

    print(f_names)
    create_boxplot(f_vals, f_names, join(folder, "box_plot_all"))
    create_boxplot(f_vals[:,:][f_classes==0], f_names, join(folder, "box_plot_not_copy"))
    create_boxplot(f_vals[:,:][f_classes==1], f_names, join(folder, "box_plot_copy"))
    pca_trans, eig_tuples = pca(f_vals)
    print(eig_tuples)
    scatter_3d(pca_trans, f_classes, join(folder, "pca_3d"))
    # tsne_trans = tsne(f_vals)
    # scatter_3d(tsne_trans, f_classes, join(folder, "tsne_3d"))

    # print("Initiating Neural Network")
    # network = getNN(input_dim=len(f_vals[0]), hidden_dims=hidden_dims)
    # print("Starting training.")
    # network.k_fold_crossvalidation(f_vals, f_classes, files, f_names,10,join(folder, "cross_results"), num_epochs, conf_thresh)
    # print("Training done!")

    # print("Initiating Neural Network")
    # network = getNN(input_dim=len(f_vals[0]), hidden_dims=hidden_dims)
    # print("Starting training.")
    # network.k_fold_crossvalidation(red_vals, f_classes, files, red_names,10,join(folder, "cross_results_reduced"), num_epochs, conf_thresh)
    # print("Training done!")

	# # Do some longer analysis
	# plot(nodes=True, epochs=True, bias=True, features, classes)

	# from log_reg import Log_Reg
	
	# print("Initiating Logistic Regression")
	# lg = Log_Reg()
	# print("Starting training.")
	# lg.kfold_log_reg(f_vals,np.array(f_classes), files)
	# print("Starting Evaluation!")


