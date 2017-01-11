# Classification.py

import os, sys
from os.path import join, realpath, dirname, isdir, basename

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties

# import for pathes for important folders
from doc_globals import* 

# pretty table visualization help
from prettytable import PrettyTable

# import classification methods
from feature_classifier import NN, Log_Reg

### Data Processing ###
def gen_train_test_split(document_ids, test_size):
    '''
    Splits th provided data into training- and testsplits

    @param  document_ids: The document ids of the classified data
    @type   document_ids: list(string)

    @param  test_size: The fraction of the data supposed to be testing-data
    @type   test_size: float
    '''
    n_data = len(document_ids)
    n_test = int(n_data*test_size)
    doc_copy = document_ids[:]
    random.shuffle(doc_copy)
    train_doc = doc_copy[:n_test]
    test_doc = doc_copy[n_test:]

    return train_doc, test_doc

def load_data(data_file, error_feature=True, rep_nan=True, norm=True):
    '''
    Loads data from a csv file. The second last row is expected to contain the classification category and the last row the filename/document_id. 

    @param  data_file: The path to the data
    @type   data_file: str

    @param  error_feature: Flag specifying if the error feature is to be added
    @type   error_feature: boolean

    @param  rep_nan: Flag specifying if nans are to be replaced
    @type   rep_nan: boolean

    @param  norm: Flag specifying if the data s to be normalized
    @type   norm: boolean

    @return np_features: The feature value matrix from the csv file
    @rtype  np_features: np.array(float)

    @return np_classes: The classifications for the feature vectors
    @rtype  np_classes: np.array(int)

    @return filenames: The filenames/document_ids for the vectors
    @rtype  filenames: list(str)

    @return column_names: The column for the data matrix only
    @rtype  column_names: list(str)
    '''
    # load the data into a pandas dataframe
    features=pd.read_csv(data_file, header=0, delimiter=',', quoting=1, encoding='utf-8')
    
    # get the headers
    column_names = list(features)
    # get tghe data
    np_features = features.as_matrix(column_names[:-2])
    # get classification
    np_classes = np.array(features[column_names[-2]].tolist())
    # get the document_ids
    filenames = features[column_names[-1]].tolist()
    # cut away doc_id and classification column name
    column_names = column_names[:-2]

    # add an error column that adds contains a one if there is a np.nan in the a data row or a zero otherwise
    if(error_feature):
        np_features = add_error_feature(np_features)
        # add error features column name
        column_names.append("error")
    # replace nans by the columns mean
    if(rep_nan):
        np_features = replace_nan_mean(np_features)
    # normalize the data
    if(norm):
        np_features = norm_features(np_features)


    return np_features, np_classes, filenames, column_names
    
def add_error_feature(features):
    '''
    Checks for np.nan in the feature matrix rows and adds another boolean error feature

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The feature matrix with the additional error feature
    @rtype   features: np.array
    '''
    error_feature = np.zeros((len(features),1))
    
    # Go through every row of data
    for i in range(0,len(features)):
        for x in np.float64(features[i,:]):
            if(np.isnan(x)):
                # If a np.nan was found somewhere in a row add a 1 at the index and go to the next row
                error_feature[i] = 1.0
                break
    features = np.append(features, error_feature, axis=1)
    return features

def replace_nan_mean(features):
    '''
    Replaces np.nan in the feature matrix with the mean of the respective feature

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The corrected feature matrix
    @rtype   features: np.array
    '''
    # Make sure that every field in the data is of type np.float64
    features = np.array(features)
    features = features.astype(np.float64)

    # compute the mean of each column excluding the nans
    col_mean = np.nanmean(features,axis=0)
    # get positions of nans
    inds = np.where(np.isnan(features))
    # repalce them with the respective mean
    features[inds]=np.take(col_mean,inds[1])

    # # Another method to do that
    # features = np.where(np.isnan(features), np.ma.array(features, mask=np.isnan(features)).mean(axis=0), features)
    return features

def replace_nan_weighted_dist(features):
    '''
    Replaces np.nan in the feature matrix by a weighted mean of the feature. IN taking the mean of a feature the value of a row is scaled by the distance to the row where the nan is supposed to be replaced. So the mean has to be recomputed for every missing nan.

    @param  features:The feature matrix
    @type   features: np.array

    @return  features: The corrected feature matrix
    @rtype   features: np.array
    '''
    #TODO: For now just take the simply mean method
    return replace_nan_mean(features)

def norm_features(features):
    '''
    Normalize the feature matrix. Make sure to handle np.nans beforehand

    @param  features: The feature matrix
    @type   features: np.array

    @return  features: The normalized matrix
    @rtype   features: np.array
    '''
    len_feat = len(features[0])
    max_nor=np.amax(features, axis=0)
    min_nor=np.amin(features, axis=0)
    for i in range(0, len_feat):
        f_range = (max_nor[i]-min_nor[i])
        if(f_range>0):
            features[:,i] = (features[:,i]-min_nor[i])/f_range
        else:
            print("The feature at position %s has always the same value!"%(i,))
    return features

def pca(data, dims=3):
    '''
    Do PCA on the data

    @param  data: The data matrix
    @type   data: np.array

    @param  dims: The number of dimensions of the transformed matrix
    @type   dims: int

    @return  data_trans: The transformed matrix
    @rtype   data_trans: np.array

    @return  eig_pairs: The eigenvectors ordered according to their eigenvalue's magnitude
    @rtype   eig_pairs: list(tuple(float,np.array))
    '''
    (n, d) = data.shape;
    data = data - np.tile(np.mean(data, 0), (n, 1));
    cov = np.dot(data.T, data)/(n-1)

    # create a list of pair (eigenvalue, eigenvector) tuples
    eig_val, eig_vec = np.linalg.eig(cov)
    # get the sum of the eigenvalues for normalization
    sum_eig_vals = np.sum(eig_val)
    eig_pairs = []
    for x in range(0,len(eig_val)):
        eig_pairs.append((np.abs(eig_val[x])/sum_eig_vals,  np.real(eig_vec[:,x])))
    # sort the list starting with the highest eigenvalue
    eig_pairs.sort(key=lambda tup: tup[0], reverse=True)
    
    # get the transformation matrix by stacking the eigenvectors         
    M = np.hstack((eig_pairs[i][1].reshape(d,1) for i in range(0,dims)))

    # compute the transformed matrix
    data_trans = np.dot(data, M)
    return data_trans, eig_pairs

def test_bow_only(features_file):
    '''
    Test the accuracy of the bow features alone. 

    @param  features_file: The path to the feature file
    @type   features_file: str
    '''
    # load data
    f_vals, f_classes, files, f_names = load_data(features_file, False, False, False)

    # generate the probability based on the bow only
    # that means sum up values higher than 0.5 and lower than 0.5 respectively and substract those vectors
    # give the resulting vector probablities according which of the two categories occured more often
    probs = np.sum(f_vals[:,0:5]>0.5,axis=1)-np.sum(f_vals[:,0:5]<0.5,axis=1)
    probs = probs.astype(float)
    # the one side has to be at least 2 more indicators than the other side to be predicted as that side
    probs[probs>1]=1.0
    probs[probs==0]=0.5
    probs[probs==1]=0.5
    probs[probs==-1]=0.5
    probs[probs<-1]=0.0
    # assign the prediction
    preds = probs>=0.5

    # concatenate relevant values prediction and real classification vectors
    inter = np.concatenate((np.concatenate((f_vals[:,0:5],preds.reshape(len(f_vals),1)),axis=1),f_classes.reshape(len(f_vals),1)),axis=1)
    # print those rowes where a classification has been made and it was wrong
    for i in range(len(f_vals)):
        if(preds[i]!=f_classes[i] and probs[i]!=0.5):
            print(inter[i,:])

    # Get the overall scores
    score = float(np.sum([preds==f_classes]))/len(f_classes)
    print("accuracy: %.4f" %(score,))
    pos_dev = np.sum(np.abs(probs-0.5)[preds==f_classes])/np.sum([preds==f_classes])
    neg_dev = np.sum(np.abs(probs-0.5)[preds!=f_classes])/np.sum([preds!=f_classes])
    print("correct_std_0.5: %.4f" %(pos_dev,))
    print("flase_std_0.5: %.4f" %(neg_dev,))
    # Get the scores for only the classified part
    preds = probs[probs!=0.5]
    f_classes2 = f_classes[probs!=0.5]
    preds = preds>=0.5
    score2 = float(np.sum([preds==f_classes2]))/len(f_classes2)
    print("accuracy2: %.4f" %(score2,))
    print("predicted: %d/%d" %(len(f_classes2),len(f_classes)))
    print('\n')


### Visalizations ###
def scatter_3d(data, classes, filepath):
    '''
    Scatter plot 3d classification-data according to their classification

    @param  data: The data matrix
    @type   data: np.array

    @param  classes: The classification for each data row
    @type   classes: np.array

    @param  filepath: The path where to store pictures of the plot
    @type   filepath: str
    '''
    # Make sure the data s of the rght dimensions
    (n,d) = np.shape(data)
    assert d == 3 , 'Need to have 3 dimensions'
    # create a figure
    fig = plt.figure('scatter3D')
    # set the resolution to high definition
    dpi = fig.get_dpi()
    fig.set_size_inches(1920.0/float(dpi),1080.0/float(dpi))
    # create a 3d plot
    ax = fig.add_subplot(111, projection='3d')
    
    # arrange the ticks nicely
    maxTicks = 5
    loc = plt.MaxNLocator(maxTicks)
    ax.yaxis.set_major_locator(loc)
    ax.zaxis.set_major_locator(loc)
    
    # add labels
    ax.set_title('scatter3D')
    ax.set_xlabel('\n' +'Dim1', linespacing=1.5)
    ax.set_ylabel('\n' +'Dim2', linespacing=1.5)
    ax.set_zlabel('\n' +'Dim3', linespacing=1.5)
    
    # scatter the data in using different colors for each class
    t_data = data[:,:][classes==1]
    f_data = data[:,:][classes==0]
    ax.scatter(t_data[:,0],t_data[:,1],t_data[:,2],s=10, label = "protected", edgecolor=colors[0])
    ax.scatter(f_data[:,0],f_data[:,1],f_data[:,2],s=10, label = "not_protected", edgecolor=colors[1])
         
    # add legend with specific Font properties
    fontP = FontProperties()
    fontP.set_size('medium')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0,0.7), prop=fontP)
    ax.grid('on')
    
    # set a viewing angle for the 3d picture
    ax.view_init(elev=10, azim=-90)

    # save the plot as figures from different angles
    fig.savefig(filepath+"_e10a90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=80, azim=-90)
    fig.savefig(filepath+"_e80a-90", bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.view_init(elev=15, azim=-15)
    fig.savefig(filepath+"_e15a-15", bbox_extra_artists=(lgd,), bbox_inches='tight')

    # show the 3d plot
    plt.show()

def create_boxplot(data, collumn_names, filepath):
    '''
    Create boxplots for each data column.

    @param  data: The data matrix
    @type   data: np.array

    @param  collumn_names: The name of a column which will be the label of the according boxplot
    @type   collumn_names: list(str)

    @param  filepath: The path where to store pictures of the plot
    @type   filepath: str
    '''
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

def print_pca_eigenvectors(eig_tuples, column_names, filename):
    '''
    Prints usefull information about the eigenvectors after a pca analysis. High eigenvalues mean that the respective vector explaines a lot of the variance of the data. The numbers in a vector indicate the magnitude of the respective feature that is contributing to the vectors direction. The data needs to be normalized fot this to make any sense. 

    @param eig_pairs: The eigenvectors ordered according to their eigenvalue's magnitude
    @type  eig_pairs: list(tuple(float,np.array))

    @param  column_names: The names of a column which will be the label of the column
    @type   column_names: list(str)

    @param  filename: The path to the file where to write the analysis
    @type   filename: str
    '''
    # open the file
    with open(filename, 'w') as f:

        # create a legend of the column names
        legend = ""
        for i,name in enumerate(column_names):
            legend += "%02d"%(i+1,)+":"+"%-20s"%(name,)+"\t"
            if((i+1)%4==0):
                legend+="\n"

        # write some header
        f.write("PCA Eigenvectors".center(80) + "\n\n" + "Legend:".center(80) + "\n")
        # write the legend
        f.write(legend + "\n\n")

        # write each vector
        for row in eig_tuples:
            eig_vec = row[1]
            tmp = [(i+1,val) for (i,val) in enumerate(eig_vec)]
            tmp.sort(key=lambda tup: np.abs(tup[1]), reverse=True)  
            vec_string = ""
            for i,entry in enumerate(tmp):
                vec_string += "%02d"%(entry[0],)+" : "+"%.3f"%(entry[1],)+"\t"
                if((i+1)%4==0):
                    vec_string+="\n"
            # write the explained varaince
            f.write(("explained_variance:%.3f"%(row[0],)).center(80) + "\n")
            # write the vector values
            f.write(vec_string + "\n\n")

if __name__ == "__main__":
    args = sys.argv
    len_args = len(args)

    usage = '''Usage:
    python3 Classification.py <feature_file.csv> [<second_feature_file.csv>]
    \t- Crossvalidation if just one file,
    \t- Training on the first and evaluation on the Second for two files.

    * Make sure all files contain the same column headers and a classification column in the second last column and a document_id column as last one!!

    ** You probably want to do other stuff than the default so have a closer look at the main and the other function provided in this script...'''

    # Check for the correct amount of arguments
    if(len_args>3 or len_args<2):
        print("Error: The number of arguments is wrong!!")
        print(usage)
        sys.exit(1)

    # Load the first feature file
    features_file = args[1]
    if not os.path.isfile(features_file):
        print("Error: Features file %s doesn't exist."%(features_file,))
        exit();

    # get its name
    feature_file_name = basename(features_file).split(".")[0]
    # get the values
    f_vals, f_classes, files, f_names = load_data(features_file)

    folder = join(RESULT_PATH, feature_file_name)
    if(not(isdir(folder))):
        os.mkdir(folder)

    # # Exclude some features 
    # f_vals = f_vals[:,[0,-2]]
    # f_names = f_names[0,-2]
    # folder = join(RESULT_PATH, "exc_last_feature")

    # Initialize parameters
    hidden_layers = 1
    hidden_dims = 500
    num_epochs = 500
    conf_thresh = 0.5

    # Specify a file for evaluation statistics
    trial_name = "NN_hl"+str(hidden_layers)+"_hd"+str(hidden_dims)+"_ne"+str(num_epochs)+"_t"+str(conf_thresh)
    trial_folder = join(folder,trial_name)
    if(not(isdir(trial_folder))):
        os.mkdir(trial_folder)

    if(len_args==2):

        # Start crossvalidation
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
        print("Crossvalidation done!")

        # # Use logistic regression
        # print("Initiating Logistic Regression")
        # lg = Log_Reg()
        # lg.kfold_log_reg(f_vals,np.array(f_classes), files)
        # print("Crossvalidation done!")

        # # Test the bow features only
        # test_bow_only(args[1])

    else:
        # Load the testing file
        test_file = args[2]
        t_f_vals, t_f_classes, t_files, t_f_names = load_data(test_file)

        # # Exclude some features 
        # t_f_vals  = t_f_vals[:,[0,-2]]
        # t_f_names = t_f_names[0,-2]

        # Train and test the features with the respective data
        print("Initiating Neural Network")
        network = NN(len(f_vals[0]), hidden_dims, hidden_layers, trial_folder)
        network.train_testNN(f_vals, f_classes, t_f_vals, t_f_classes, num_epochs)

        # # Use logistic regression
        # print("Initiating Logistic Regression")
        # lg = Log_Reg()
        # lg.train_test(f_vals, f_classes, t_f_vals, t_f_classes)

    # # Create boxplots for each column in the feature data
    # create_boxplot(f_vals, f_names, join(folder, "feature_box_plot_all.png"))
    # create_boxplot(f_vals[:,:][f_classes==0], f_names, join(folder, "feature_box_plot_not_copy.png"))
    # create_boxplot(f_vals[:,:][f_classes==1], f_names, join(folder, "feature_box_plot_copy.png"))
    
    # # Do PCA to have a look at the feature space
    # pca_trans, eig_tuples = pca(f_vals)
    # write_pca_eigenvectors(eig_tuples, f_names, join(folder, "eigenvectors_pca.txt"))
    # scatter_3d(pca_trans, f_classes, join(folder, "pca_3d"))



