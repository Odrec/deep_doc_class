# Classification.py

import os, sys
from os.path import join, realpath, dirname, isdir, basename

import csv
import numpy as np
import pandas as pd
from doc_globals import* 

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties

from prettytable import PrettyTable

def load_data(features_file):
    #loads the features from the csv file created during extraction
    #
    #@result:   list of lists of features, list of corresponding classes and filenames
    features=pd.read_csv(features_file, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    f_names = list(features)
    np_features = features.as_matrix(f_names[:-2])
    np_classes = np.array(features[f_names[-2]].tolist())
    filenames = features[f_names[-1]].tolist()

    error_col, error_names = generate_error_features(np_features)
    np_features = np.append(np_features, error_col, axis=1)
    f_names = f_names[:-2]
    f_names.extend(error_names)

    return np_features, np_classes, filenames, f_names[:-2]
    
def generate_error_features(features):
    #generates the error features and replaces the nan values
    #
    #@result:   list of features with the error features included
    error_feature = np.zeros((len(features),1))
    
    for i in range(0,len(features)):
        for x in np.float64(features[i,:]):
            if(np.isnan(x)):
                error_feature[i] = 1.0
                break
    
    return error_feature, ["error_mod"]

def replace_nan_mean(features):
    for x in features:
       for i, a in enumerate(x):
           x[i] = np.float64(a)
    # features = np.array(features)
    # features = np.where(np.isnan(features), np.ma.array(features, mask=np.isnan(features)).mean(axis=0), features)
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

def pca(data, dims=3):
    (n, d) = data.shape;
    data = data - np.tile(np.mean(data, 0), (n, 1));
    cov = np.dot(data.T, data)/(n-1)
    # create a list of pair (eigenvalue, eigenvector) tuples
    eig_val, eig_vec = np.linalg.eig(cov)
    sum_eig_vals = np.sum(eig_val)
    eig_pairs = []
    for x in range(0,len(eig_val)):
        eig_pairs.append((np.abs(eig_val[x])/sum_eig_vals,  np.real(eig_vec[:,x])))
    
    # sort the list starting with the highest eigenvalue
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)          
    M = np.hstack((eig_pairs[i][1].reshape(d,1) for i in range(0,dims)))

    data_trans = np.dot(data, M)
    return data_trans, eig_pairs

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
    
    t_data = data[:,:][classes==1]
    n_data = data[:,:][classes==0]
    ax.scatter(t_data[:,0],t_data[:,1],t_data[:,2],s=10, label = "protected", edgecolor=colors[0])
    ax.scatter(n_data[:,0],n_data[:,1],n_data[:,2],s=10, label = "not_protected", edgecolor=colors[1])
         
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

def write_pca_eigenvectors(eig_tuples,f_names, filename):
    with open(filename, 'w') as f:

        legend = ""
        for i,name in enumerate(f_names):
            legend += "%02d"%(i+1,)+":"+"%-20s"%(name,)+"\t"
            if((i+1)%4==0):
                legend+="\n"

        f.write("PCA Eigenvectors".center(80))
        f.write("\n")
        f.write("\n")
        f.write("Legend:".center(80))
        f.write("\n")
        f.write(legend)
        f.write("\n\n")
        for row in eig_tuples:
            eig_vec = row[1]
            tmp = [(i+1,val) for (i,val) in enumerate(eig_vec)]
            tmp.sort(key=lambda tup: np.abs(tup[1]), reverse=True)  
            vec_string = ""
            for i,entry in enumerate(tmp):
                vec_string += "%02d"%(entry[0],)+" : "+"%.3f"%(entry[1],)+"\t"
                if((i+1)%4==0):
                    vec_string+="\n"
            f.write(("explained_variance:%.3f"%(row[0],)).center(80))
            f.write("\n")
            f.write(vec_string)
            f.write("\n\n")

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

    feature_file_name = basename(features_file).split(".")[0]

    f_vals, f_classes, files, f_names = load_data(features_file)

    f_vals = replace_nan_mean(f_vals)
    f_vals = norm_features(f_vals)

    folder = join(RESULT_PATH, feature_file_name)
    if(not(isdir(folder))):
        os.mkdir(folder)

    # # Exclude some features 
    # meta = [9,10,11,12,13,14]
    # others = [1,2,3,4,5,6,7,8,15,16,17,18,19,20,21,22,23,24,25,26]
    # f_vals = f_vals[:,meta]
    # f_names = [f_names[i] for i in meta]
    # folder = join(RESULT_PATH, "only_meta_csv")

    # create_boxplot(f_vals, f_names, join(folder, "feature_box_plot_all.png"))
    # create_boxplot(f_vals[:,:][f_classes==0], f_names, join(folder, "feature_box_plot_not_copy.png"))
    # create_boxplot(f_vals[:,:][f_classes==1], f_names, join(folder, "feature_box_plot_copy.png"))
    # pca_trans, eig_tuples = pca(f_vals)
    # write_pca_eigenvectors(eig_tuples, f_names, join(folder, "eigenvectors_pca.txt"))
    # scatter_3d(pca_trans, f_classes, join(folder, "pca_3d"))

    from simple_neural_network import NN
    hidden_layers = 1
    hidden_dims = 500
    num_epochs = 500
    conf_thresh = 0.5

    trial_name = "NN_hl"+str(hidden_layers)+"_hd"+str(hidden_dims)+"_ne"+str(num_epochs)+"_t"+str(conf_thresh)
    trial_folder = join(folder,trial_name)
    if(not(isdir(trial_folder))):
        os.mkdir(trial_folder)

    print("Initiating Neural Network")
    network = NN(len(f_vals[0]), hidden_dims, hidden_layers, trial_folder)

    network.k_fold_crossvalidation(f_vals,
        f_classes,
        files,
        f_names,
        10,
        trial_folder,
        num_epochs,
        conf_thresh)
    print("Training done!")

    # from log_reg import Log_Reg
    # print("Initiating Logistic Regression")
    # lg = Log_Reg()
    # print("Starting training.")
    # lg.kfold_log_reg(f_vals,np.array(f_classes), files)
    # print("Starting Evaluation!")


