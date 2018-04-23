# classify_crp.py
import os, sys
from os.path import join, realpath, dirname, isdir, basename, splitext
import itertools
# import for pathes for important folders

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)
from Feature_Extractor import Feature_Extractor

import csv, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axis3d as a3d #@UnresolvedImport
from matplotlib.font_manager import FontProperties

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score

from copy import deepcopy
from itertools import compress

# visualization help
from prettytable import PrettyTable
import colorsys

from Classifier import Classifier

### general helper functions ###
def pause():
    """
    Pause the execution until user presses a enter
    """
    input("Press Enter to continue...")
    return

### specific deep_doc_class helper functions ####
def setup_feature_extractor():
    '''
    Creates a Feature_Extractor with alle current  pathes

    @return fe: Featrue_Extractor obejct with all feature pathes set up
    @rtype  fe: Featrue_Extractor
    '''
    # set pathes to the data folders
    DATA_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/"
    PRE_EXTRACTED_DATA_PATH = join(DATA_PATH, "pre_extracted_data")

    # get pathes to the needed files
    prop_dir = join(PRE_EXTRACTED_DATA_PATH,"pdf_properties_new.json")
    struc_dir = join(PRE_EXTRACTED_DATA_PATH,"pdf_structure.json")
    meta_dir = join(DATA_PATH,"classified_metadata.csv")
    text_dir = join(DATA_PATH,"xml_text_files")

    # load json or csv data
    metadata=pd.read_csv(meta_dir, delimiter=',', quoting=1, encoding='utf-8')
    with open(prop_dir, "r") as prop_file:
        properties = json.load(prop_file)
    with open(struc_dir, "r") as struc_file:
        xml_structure = json.load(struc_file)

    # return Feature_Extractor object
    fe = Feature_Extractor(
        text_dir=text_dir,
        metadata=metadata,
        properties=properties,
        structure=xml_structure)
    return fe

def load_data(data_file, delimiter=",", rep_nan="mean", norm=True):
    '''
    Loads data from a csv file. The first column is expected to contain the document_id and the second column the classification category.

    @param  data_file: The path to the data
    @type   data_file: str

    @param  rep_nan: For now either "mean" or None
    @type   rep_nan: str

    @param  delimiter: Delimiter of th csv file
    @type   delimiter: str

    @param  norm: Flag specifying if the data s to be normalized
    @type   norm: boolean

    @return data: The feature value matrix from the csv file
    @rtype  data: np.array(float)

    @return classes: The classifications for the feature vectors
    @rtype  classes: np.array(int)

    @return ids: The ids/document_ids for the vectors
    @rtype  ids: list(str)

    @return column_names: The column for the data matrix only
    @rtype  column_names: list(str)
    '''
    # load the data into a pandas dataframe
    features=pd.read_csv(data_file, header=0, delimiter=',', quoting=1, encoding='utf-8')

    # get the headers
    column_names = list(features)
    # get tghe data
    data = features.as_matrix(column_names[2:])
    # get classification
    classes = np.array(features[column_names[1]].tolist())
    # get the document_ids
    ids = features[column_names[0]].tolist()
    # cut away doc_id and classification column name
    column_names = column_names[2:]

    # replace nans by the columns mean
    if(rep_nan):
        data = replace_nan_mean(data)
    # normalize the data
    if(norm):
        data = norm_features(data)


    return data, classes, ids, column_names

### PARAMETER OPTIMIZATION ###
def _get_result_row(classifier, train_data, train_labels, test_data, test_labels):
    results_row = []
    # train set
    probs = classifier.predict_proba(train_data)[:,1]
    preds = probs>=0.5
    results_row.append(accuracy_score(train_labels, preds))
    # test set
    probs = classifier.predict_proba(test_data)[:,1]
    preds = probs>=0.5
    # add results
    results_row.append(accuracy_score(test_labels, preds))
    if(not(np.sum(preds)==0)):
        results_row.append(precision_score(test_labels, preds))
    else:
        results_row.append(0)
    results_row.append(recall_score(test_labels, preds))
    results_row.append(fbeta_score(test_labels, preds, 1.7))
    results_row.extend(confusion_matrix(test_labels, preds).ravel()[1:3])

    # format results
    results_row[0:5] = ["%.4f"%(val,) for val in results_row[0:5]]
    results_row[5] = "%d   %.3f"%(results_row[5],results_row[5]/np.sum(test_labels==0))
    results_row[6] = "%d   %.3f"%(results_row[6],results_row[6]/np.sum(test_labels==1))
    return results_row

def _save_search_results(rows, sort_by, res_header, model_header, results_file):
    res_tab = PrettyTable(["id"] + res_header)
    model_tab = PrettyTable(["id"] + model_header)

    res_sort = sorted(rows, key=lambda x: x[res_header.index(sort_by)+2], reverse=True)
    for i,r in enumerate(res_sort):
        res_tab.add_row([i+1]+list(r[2:]))
        model_tab.add_row([i+1]+[r[0][k] for k in model_header])

    # write results to file
    with open(results_file, 'w') as f:
        f.write(("Grid Parameter Combination Results:").center(80) + "\n\n")
        f.write(res_tab.get_string())
        f.write("\n\nModel Parameter\n")
        f.write(model_tab.get_string())

def random_search_bin(classifier,
    n_models,
    params,
    train_data,
    train_labels,
    test_data,
    test_labels,
    sort_by="acc",
    use_deepcopy=True,
    results_file=None):
    header = ["train_acc", "acc", "precision", "recall", "f_beta", "fp", "fn"]
    p_keys = list(params.keys())
    rows = []

    p_items = params.items()
    for i in range(n_models):
        args = {}
        for k in p_keys:
            p_gen = params[k]
            if hasattr(p_gen, "rvs"):
                args[k] = p_gen.rvs()
            elif(callable(p_gen)):
                args[k] = p_gen()
            else:
                args[k] = np.random.choice(p_gen)
            if(type(args[k])==np.str_):
                args[k] = str(args[k])

        classifier.set_params(**args)
        classifier.fit(X=train_data, y=train_labels)

        results_row = _get_result_row(classifier, train_data, train_labels, test_data, test_labels)

        if(use_deepcopy):
            rows.append((args,deepcopy(classifier)) + tuple(results_row))
        else:
            rows.append((args,classifier.get_deepcopy()) + tuple(results_row))

    if(not(results_file is None)):
        _save_search_results(rows, sort_by, header, p_keys, results_file)

    return rows, header

def grid_search_bin(classifier,
    params,
    train_data,
    train_labels,
    test_data,
    test_labels,
    sort_by="acc",
    use_deepcopy=True,
    results_file=None):

    header = ["train_acc", "acc", "precision", "recall", "f_beta", "fp", "fn"]
    rows = []
    p_keys = list(params.keys())
    comb_param_vals = list(params.values())
    combinations = list(itertools.product(*comb_param_vals))

    for i in range(len(combinations)):
        args = dict(zip(p_keys, combinations[i]))
        classifier.set_params(**args)
        classifier.fit(X=train_data, y=train_labels)

        results_row = _get_result_row(classifier, train_data, train_labels, test_data, test_labels)

        if(use_deepcopy):
            rows.append((args,deepcopy(classifier)) + tuple(results_row))
        else:
            rows.append((args,classifier.get_deepcopy()) + tuple(results_row))

    if(not(results_file is None)):
        _save_search_results(rows, sort_by, header, p_keys, results_file)

    return rows, header

### Classification ###
def crossvalidate_proba_bin(model, data, labels, kfolds=10, shuffle=True, seed=None):

    # the seed makes sure to get the same random distribution every time
    if(not(seed is None)):
        np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=kfolds, shuffle=shuffle, random_state=seed)

    # count the itrations
    kfold_iter = 0

    train_results = np.ones((len(data),kfolds))*-1
    test_results = np.ones((len(data),kfolds))*-1
    # test_results = np.zeros((len(data),2))

    # evaluate each split
    for train, test in kfold.split(data,labels):

        model.re_initialize()
        # split the data
        train_data = data[train]
        test_data = data[test]
        train_labels = labels[train]
        test_labels = labels[test]

        # train the model
        model.train(train_data, train_labels)

        # predict the training set
        train_probs = model.predict_proba_binary(train_data)[:]
        train_results[train,kfold_iter] = train_probs

        # predict the testing set
        test_probs = model.predict_proba_binary(test_data)
        test_results[test,kfold_iter] = test_probs
        # test_results[test,0] = test_probs
        # test_results[test,1] = kfold_iter
        kfold_iter += 1

    return train_results, test_results

def ddc_cat_crossvalidation_bin(model, doc_ids, labels, results_path, kfolds=10, shuffle=True, seed=None):

    # get a Feature_Extractor
    fe = setup_feature_extractor()
    # get pathes to train and test files
    DATA_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/"
    train_file = join(DATA_PATH, "feature_values", "cross_eval_train.csv")
    test_file = join(DATA_PATH, "feature_values", "cross_eval_test.csv")

    # the seed makes sure to get the same random distribution every time
    if(not(seed is None)):
        np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=kfolds, shuffle=shuffle, random_state=seed)

    # count the itrations
    kfold_iter = 0
    # reserve space for results
    train_results = np.ones((len(doc_ids),kfolds))*-1
    test_results = np.ones((len(doc_ids),kfolds))*-1

    # evaluate each split
    for train, test in kfold.split(doc_ids,labels):
        # make sure the model is untrained
        model.re_initialize()
        # get train and test document ids
        train_ids = [doc_ids[i] for i in train]
        test_ids = [doc_ids[i] for i in test]
        train_labels = labels[train]
        test_labels = labels[test]

        # let the feature extractor train the bow classifiers on the data only
        fe.train_bow_classifier(train_ids, train_labels, "custom", "log_reg")

        #  extract training and testing data
        fe.extract_features(doc_input=zip(train_ids,train_labels) ,feature_file=train_file)
        fe.extract_features(doc_input=zip(test_ids,test_labels) ,feature_file=test_file)

        # load that data from file
        train_data, train_labels, train_docs, column_names = load_data(train_file)
        test_data, test_labels, test_docs, column_names = load_data(test_file)

        # train the model
        model.train(train_data, train_labels)
        # get a ranking of the feature importance and save a txt
        analyse_feature_ranking(classifier=model, feature_names=column_names, results_path=join(results_path,"feature_ranking_%d.txt"%(kfold_iter,)))

        # predict the training set
        train_probs = model.predict_proba_binary(train_data)[:]
        train_results[train,kfold_iter] = train_probs

        # predict the testing set
        test_probs = model.predict_proba_binary(test_data)
        test_results[test,kfold_iter] = test_probs

        kfold_iter += 1
        print("%d. iteration done!%"(kfold_iter, ))

    return train_results, test_results

### Results analysis ###
def analyse_crossvalidation_results_bin(results, labels, results_path, filename, thres=[0.25,0.5,0.75], boxplots=True):

    if(not(isdir(results_path))):
        os.makedirs(results_path)

    legend = "Copyright protected documents have a positive label.\n"+\
        "Not copyright protected documents have a negative label.\n"+\
        "False positives are documents that have falsely been predicted as copyrights!\n"+\
        "False negatives are copyright protected documents which were not found!\n\n"+\
        "Precision: true positives / predicted positives\n"+\
        "Recall: true positives / real positives\n"+\
        "F1: 2* precision*recall/(precision+recall)\n\n"

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    # evaluate the results considering different thresholds
    test_size = len(labels)/len(results[0])
    for t in thres:
        # create a table
        results_table = PrettyTable(score_names)
        # get the average scores
        average_scores = np.zeros((8,))
        for k in range(0,len(results[0])):
            indices = results[:,k]>=0
            probs = results[indices,k]
            preds = probs>t
            target = labels[indices]

            results_row = []
            results_row.append(accuracy_score(target, preds))
            results_row.append(precision_score(target, preds, average="binary"))
            results_row.append(recall_score(target, preds, average="binary"))
            results_row.append(f1_score(target, preds, average="binary"))
            results_row.extend(confusion_matrix(target, preds).ravel())

            average_scores += np.array(results_row)
            results_row[0:4] = ["%.3f"%(val,) for val in results_row[0:4]]
            results_row[4:] = ["%d   %.3f"%(val,val/test_size) for val in results_row[4:]]
            results_table.add_row(results_row)

        results_table.add_row(['']*len(score_names))
        average_scores /= len(results[0])
        average_scores = list(average_scores)
        average_scores[0:4] = ["%.3f"%(val,) for val in average_scores[0:4]]
        average_scores[4:] = ["%d   %.3f"%(int(val),val/test_size) for val in average_scores[4:]]
        results_table.add_row(average_scores)

        with open(join(results_path, filename + "_cross_eval_thres_%.3f.txt"%(t,)), 'w') as f:
            f.write(("Crossvalidation results for threshold %.3f"%(t,)).center(80) + "\n")
            f.write("\n"+legend+"\n\n")
            f.write(results_table.get_string())

    if(boxplots):
        all_probs = (results.T).reshape(-1)
        all_labels = np.array(list(labels)*len(results[0]))
        create_boxplot(np.concatenate((all_pros[all_lables==0],all_pros[all_lables==0]),axis=1), ["label:0","label:1"], join(results_path, "probs_per_label.png"))

def update_cat_eval(category_eval, false_neg_ids, doc_ids, pred):

    false_docs = ["cc17fb8eaf055e2c8224eaaf3c10145c",
        "43d566f90c24f874781d97905870ed23",
        "96a92f3bcb002f22a698a62a468415e3",
        "9c70eaa380eb60135c5f1953b1f073cf",
        "6aaf86df732ef957de9a48e2b3c589e9",
        "daad7c32072aaa4cdbb9ae2c02c2c9b6",
        "e65752213cebcb6e151a815ed21512f3",
        "5ff6d40b1dc0dde4c51a8436bc57aa20",
        "fc8278775cf12fdf84e88493f8efc645",
        "d047a01a190236e10d6addefe13ed3bc",
        "7e7ff5a990f0e5756e31348ced169854",
        "5268494d6a5b7f3bc27bc2dcbcf79748",
        "10c8f8111c29862458987f77c05f4a39",
        "19e69897d4842c053d781e773e5122bd",
        "da5a198a45c690c3570a7d307597b892",
        "f51d621abeb4c489918d904995bbcec0",
        "ef59be13bd0d88b43fcb779912df7a66",
        "36889863bea6feb43dbe335719aa657a",
        "6c357bc6ddc11a3b5d199d2b1e4e6fac",
        "2ad710735830e95c3e9ce01725552a8b",
        "df59eaa1565f4b3d980c15a00ade3bdc",
        "826c1f6b7d9b2bd7fc50c23a126b71a1",
        "12d0d64889fe56c5789f8b73a8813ce9",
        "9d2cb33abdcfefa9a56bb1f7da4ef263",
        "a5322615e83b66d9b6c5e9a50b3eebb2",
        "57f98112812429c28def64a14020b242",
        "b37ce79bf563844426c7452c38564acc",
        "00a1e2413c919d6e392442533626cb92",
        "acda4e5d66d28ab616cd34fd09055b05",
        "ea6c27a337aff6c262e8316984ff72e5",
        "b2ac2fe5be5547e864f00b2d0f3d8003",
        "7f4038da9e1fb0aa745adcf8cc8cb832",
        "e5b0ee8a5519969ec70f5722a90a8019",
        "a774df02a832585275de3db40f132cdd",
        "580bebc528e28f520565e69ef6de8506",
        "5464ee4ef790013f0caa5d16d72ae7fc",
        "259f6473a159a782cc6f9aaf2ceda6a1",
        "057782ecdfc1f4179d2f98c4d80da89b",
        "06f3ad19c37c9f4460b0c6482505599b",
        "0a32570101750613de0fc032b94f73f4",
        "116eee145a0276280e82e0081bc39130",
        "136d718bee74eb407355cb64f064867d",
        "14c2d5d73ebc31122a07ab587272f82e",
        "170754478dc23a930d22722b43dbf181",
        "1799aad8a5a88b17cd8480fdecdd7b81",
        "3229f885e2668f522c2d64b84a34639b",
        "3981b966fa573e6419054a38660bbdda",
        "44fbbe38e3512475b9e590a937c3e933",
        "49f920671851f6a9010187bdc9743b2c",
        "590ce47a42bf67aa1295d02e85b1a7b5",
        "5c52e2433813f16b46264f07fed83783",
        "5f4616a15b26f79b6190b072dba11aa8",
        "6956ad338f080b4be0710ce32ca12976",
        "73197946354da526a4f08756a9703ddd",
        "74f8f36f354984572718e765d4e85189",
        "8b9b83971a91e40e8a05bd09d0138e49",
        "9100a9452a85c3732784074b41233a9c",
        "a4351807fee95ad4a750546b71b657fb",
        "ad3b3d728c8e579ef31efe61b9739c8a",
        "ae5378982b7227176c3e42dd93c58bd4",
        "c7cda18baa6c465a1a076bc6034b609f",
        "d5763c8901ef9e76bbef5e673145b300",
        "ef82cdd10e45d8184c8779361cce78ef",
        "7daba79637e41d82c9df63465ba2f0a3",
        "f39e2ed04d180820f05d6ff362aedd1c",
        "9dbb018b918a13f12613a063c386e2a7",
        "bf6a3ea6f15c2c9c96a19da3f3ba9e43",
        "38700021b4e8b2b6f34e19c47a096980",
        "fabe694fea65e5b15574f4eaa2a190a9",
        "059600524e5fa4fff9c72854ae7c70dc",
        "6fa54fffe2a2335f5927572e9757ee40",
        "d9eebb93c161bb0131245f80e8ccc0b3",
        "97bb3e396c0396a693f3f8cb2f3654b3",
        "4af0c6020d07663825f905c783828379",
        "3c16342b2169cc662f7f3c4b0e9289a1"
    ]

    xml_error = [
        "4d70b0a1973a94c98b50a9fd43f0ffba",
        "462e37b58d623e5458b8d630e4a4d993",
        "e52c862fe0b09a52c94dee89141925eb",
        "168f489c98b9976bbcce28f628e5078e",
        "dab670f904564b76c3463a4343594126",
        "afe02eb0714555c8176deaab05a5342f",
        "35d1120c963d214c331c3b036377a0cf",
        "e767a496ac211ec94a7f82f905a51393",
        "b98e74c526e1ecfc9ecafe848a21e147",
        "80946afd0a6c882d243ca3d3586a150d",
        "c26eb949d5bd40ded6a0711ef9af3426",
        "aafeaf6ff8f738c537bc8b4ebf012dde",
        "7451990e707f49fc90725036440c4434",
        "0b6b1cb9c9cd832bf02cf67bf8302e2c",
        "f16934cc36ac59d568502ac3a9d34cf3",
        "a4975f6d0d4ff4b8bbd0981ecd0f2285",
        "b8e2d6b46e3cb95bb185b371c1c95e1e",
        "6b5e9e9ce2e29547726ae8f8abd2ca82",
        "477e81c2aac2ff90b327e495366f8f86",
        "07a135fa0d07c49e0e6cd47810003dbc",
        "2312cef18bcf2755f37f1e08c5ce911e",
        "4c4829b9388c25f7ec2a001902e3d648",
        "dc55cb8299cd57bfdab9c7aff817d1eb",
        "81668dba8fec354eeb91024cdd663089",
        "154e7e64bf2119f2a5129991bdf4bf7c",
        "cd977ba21b6b8e6c8f2120e2c2881474",
        "a412dd22dd26aead36dc91026405abbb",
        "7ce1765b88bd1d1a78eca1e14fff62f9",
        "35ea1cd083532790f29aa9975cd6eb45",
        "152c77685b41801858fbcc76c145dab7",
        "3fb837679d044b2cf95a52e3370c2045",
        "f4e06729807af3fd0d162790a4a48ef4",
        "9f5832a1a3eec3ef2266e523f85c4322",
        "f1dfc962076e1f7f89cdaf8efb5cb314",
        "c6237fc82c1c2030e68df38de7d3871a",
        "618452d00a628665901bdc3151265e70",
        "0a78f6074fa498a9a184b3283cde7139",
        "6888a8187c69fc8d91e45828d280bfb7",
        "5ea358146448d79442e42e15233026c9",
        "8df3f8fc7b4c6aa07a0529f441120660",
        "a9e0db397b2e8b95931bffd8a56d5c18",
        "240c7b6e495076afaba00c9889aea082",
        "7b9778f8f7fb1c67783b82fb9bae42ab",
        "e51d2152efe6856a26ddd0adc845d1f0",
        "ca3359a20a73e30b109d0aed1a4b8711",
        "b7d354d689f5052419c45cf4414d8216",
        "8bd58c19720d9cc35d119f965b1ef601",
        "26d3adda46b9c565d327d8d810e6151a",
        "6f188eac23f755fdb358a7ce1136b6e7",
        "dfae946e4721c3470994c90bd5bd393e",
        "1b06d3a26375b06e7b79afb68f80b7da",
        "d47b36df95eb67841ad803256da587de",
        "7262cec0a47a711bb4f34391557219c9",
        "d6c6bdadbbcfe0f16f31ddfa76868d5c",
        "f81c611790ded32869499b298acc9d4e",
        "4fe72e59a686fd5dcddce7e5d8d2c99e",
        "8d072fc27449b5637453d45a9d75e2b0",
        "ee31588463e2469ee6216e9c1a32186d",
        "4a1c0babb340a6bd167a2600ba25e259",
        "5aaaa751412c557f4a1230d48d61614b",
        "56a2018de85943fa30cd4ddce3b9b602",
        "f0140465484930a469c8e08be23b107b",
        "d404503b466c0067fb52ea7fbb1eb99c",
        "7dc8171d1f40cb6d4ab849e0901f9989",
        "3cd73c97de8751d51309ffcca41a2ee5",
        "3848d5e5b76f7aa570105dc0c4e23ca9",
        "4053a8bd67e6b97033b1b1e379eb07b0",
        "efe95fb04937ce282415b3752296037f",
        "998b4efbfcc024e8284aef4f0d9eba87",
        "333c25f776348ecc4c824dfe9318ffe0",
        "1104c1ccb4037a458c7f1306cc8a4b38",
        "ab9cc9945400f56e598c7642120b4ac0",
        "cff2a79f90981ec250bdc460101e92d1",
        "353618076a78e6d84e3b16607693cade",
        "32e90d1459c534087a2daccdfc98453c",
        "f38a244eb0ef01f05e8d9f88d4c15286",
        "6e6a8826597f72e5403a00f1fc43fea2",
        "f677954aab79b144b33a8539a80b5021",
        "e6e55e34f7f10acf95bf72e8f64adc9a",
        "f90fd5e8e62b6216f9575e61f3f8ef45",
        "f25e883b16833c800a297e4d93b0d553",
        "0778080d975a2f66a5bd48b249baab5f",
        "6ce159fb4aee9b927c2157086f32f925",
        "f4e6c77cc5614c657ce54a481fd0f362",
        "ec932748ed8dc56134253c82d61c101d",
        "4443af1116ccc197989be849d9f41aec",
        "2cd9220978624e281e0e3245d9d0fca5",
        "7fde22aa84dcb4b2cc9e52a480507535",
        "95cc939db7100de396da394317e28d1c",
        "e5697f2a78bc48b1eb0252fd2ef7279a",
        "de6d3ae71d8b587fb5217209fac784d8",
        "87245bf63324cf33d4aef86b13c7b0eb",
        "e95621e72ff16ef5048d3626592af65b",
        "7a623019662b3bebf5f76771051dd880",
        "97383cd50190b061553cd73973bce984",
        "f752c531b9c64ee61a33fc4452ecb976",
        "fabcd583132b6eb5d12812726f17bedb",
        "9c93f5b188319d1bddaed4171d0cf7cf",
        "f41d85f790ac6c88d3d66cd876ff0b85",
        "8d196e2c05c7e02d5117302fd7fa6182",
        "170754478dc23a930d22722b43dbf181",
        "0d6f103a9fa711786c797fb1b46e8e22",
        "6edf443e2586f9e0c91d190a9a22f2e5",
        "1ce91be84b2e5643eae607d049c2912d",
        "e3df65c8481ecd9b72f072a065bac4a6",
        "a7f54c58f810e2ea9f95ec28aad08f8b",
        "e84d829645112856c3be0b408e783115",
        "08576fc10f451db0b6161718f7ab9a8d",
        "9b42880245d1b42d5771bc2bffc3b76a",
        "6032b50d503896c3fd7d9cd31d2e0447",
        "cc375cc82ad35b12b14062163670b342",
        "b4610d02c99e9c27742441d36655f1ba",
        "0306420da9829eef44f32ea96d45d568",
        "9a791ce480ed18604f49553e7a992bab",
        "15c4bae8e5f664d9ec481ee6385f8a25",
        "74c3870963becf2b891ef2574bfcb319",
        "1a5d44eb741c693cb626198fc63b50d5",
        "f6d2282e1aa5ff337ec129a553535e67",
        "ad9ade6d7b101edcb4bb9849be69074c",
        "a2835d03b2d55cd13c406106eee9aaf1",
        "c09aaa06323524e14a734b6c711f443f",
        "01ee82f2ec05c0e273d229b5e9412379",
        "6c357bc6ddc11a3b5d199d2b1e4e6fac",
        "11f66b9e33b3773d1039ae28d45183e5",
        "1102816909e4050340e4269631e3a57e",
        "8b5d679b3ad0d2a3bf85a22c4c3fd47c",
        "835508355d5f6284a0c58a1873a90a2a",
        "c7909f905d3376d1d50e306a23a28e60",
        "48c81c9e65935473cbd4c206b023c673",
        "071e9a3c2811f70db3bac220aaa325bb",
        "843b7382bd6ec44dd2bb4c958d4a7751",
        "e4d437e678dc3c9da8b6695d3ff8d7e2",
        "9a5d3349755cdbf1fc71dbee3e36ac37",
        "68ad6882f7cd47437e58108e716cbcad",
        "9a599a19d1e277137e3f083c255a62e3",
        "ddcf3afdc941865b12bdd2af9c055ef6",
        "9cff5d5a04b718f20ca4eecc93e64d8b",
        "9b2b774ef4c39fc30384a0bf0a40f600",
        "8dbff54b538bfe88c04c33122b3bca38",
        "51acd0ea605794b421781e475e322c9a",
        "341343a4fba4b858c1c916cccf9b9cec",
        "d87cc194527e7d6c5e6a8b4037046394",
        "458c3b916b58fbd0d6efe085fb2ab2b0",
        "41d760f50056d2a3bbf086047492c436",
        "6d761eb389d6d0fddf0ee72dc384cf2e",
        "e1cef7747907cecbce8c76026eb2ca06",
        "e33e7c75475d7073160d6058053f83e6",
        "a75d7757a9c31c9dab10c2c8e4c34f89",
        "404d8dce32be7dcb2e4160aa78c44de6",
        "a8c89340f76e1798d1bb9726d91b7cb8",
        "24c9893770471a031e2c5d4619ecc995",
        "13374318ead52e16969c0d0544d7c596",
        "bb88fe6c0328912f0bb90beb4fb87109",
        "206297c4a808111bcee4d769dbb43385",
        "d987c51792fcaa51a8916b6687b09d81",
        "08199c77dbbcd19a24cf4bda154a29da",
        "5ce4ae00d419a61b987b8abfcc931b25",
        "a0240d7f82990009412be04369d5c7d1",
        "e5178ff03bb21e40281c083021dd5d80",
        "9478bb43e1060f6747290d0ef04b806c",
        "203a02d423548260c477a5c1e56c40c6",
        "66fc6ce18b44145964a06d8ff9db3ea1",
        "38d35d6dddf9aa3c5ad647563a1c7b6a",
        "56db3ea9339a2a2d8424804dd3842641",
        "c125a84aa5ac9216987a9e1f725b8e64",
        "325cc5f8fea90ec6455c1fc90805a801",
        "20ed88098c997adade05ebcc5855c20c",
        "e63baff797c0f1755888a2cbb22c4935",
        "f264c2d15410ab35c314b679674c0109",
        "008bd1a8b44ff03dc160552cf0514a92",
        "62c206e1238acf99b33906437504996f",
        "3e039943e16f3eddea67fa71335fe052",
        "70fa622e70974dbf3314adfa4ad68514",
        "c2a992d1b181ce386a342e6324131ac3",
        "9fca5e8d73098cb0bbd103ae77b16030",
        "c35ec2d4552a950069b390101bbbd27d",
        "21f0fb0e872f9cc83ebb69f0890fe1c9",
        "8441e472c36d966dc9d3f0ed9ed767ba",
        "f1abde355dac04bbd21d33574de0c299",
        "664c7b944a1799b494001ebcffd1a312",
        "24549c324d4138b4aabe2e69da3143b9",
        "0880622c454aa613513589c9957b85a9",
        "6487721de7c5bd3f50257e8931020606",
        "14a14bbc72c5ad5c4ef75714fc27a2ca",
        "f2dadc933186ba967ce56853e9242bc5",
        "f7ba2b00060e70cede9eda22f8323e7f",
        "85bf12bd6c668033a3d730f3926b0640",
        "ec5951bec760e3d149092f0cadd71119",
        "7c1b592a0b256f3bb61e42c6a0260022",
        "eee07c5ef8c9d49c5cbb4ae689b76179",
        "a69acaacc655eeadd3f726c1f2fe4dd1",
        "18111af931025656bdf4581bd27b268e",
        "c9e4d8556c070d903cb8b58cfa57e90a",
        "46fe2f38e7e10be55cecc7e87599e88a"
    ]

    cat_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/cleaned_manual_class.csv"
    cat_dict = {}

    with open(cat_path,"r") as cat_file:
        reader = csv.DictReader(cat_file)
        for row in reader:
            cat_dict[row["doc_id"]]=row

    for i in range(len(doc_ids)):
        d_id = doc_ids[i]
        # if(d_id in false_docs):
        #     continue
        label = int(cat_dict[d_id]["class"])
        cat = int(cat_dict[d_id]["category"])-1
        imp = int(cat_dict[d_id]["important"])
        p = pred[i]
        category_eval[cat,0]+=1
        if(label and not(p)):
            false_neg_ids[cat+1].append([d_id,0])
            category_eval[cat,1] +=1
            if(not(imp)):
                category_eval[cat,3]+=1
            if(d_id in xml_error or d_id in false_docs):
                false_neg_ids[cat+1][-1][1]=1
                category_eval[cat,4]+=1
        elif(not(label) and p):
            category_eval[cat,2] +=1
            if(not(imp)):
                category_eval[cat,3]+=1
            if(d_id in xml_error):
                category_eval[cat,4]+=1
        else:
            continue

    return category_eval

def analyse_crossvalidation_results_doc_categories(results, labels, doc_ids, results_path, filename, thres=[0.1,0.2,0.3,0.4,0.5], boxplots=True):

    # id description dict
    cat_id_to_des = {
        1:"scanned",
        2:"scanned_max2p",
        3:"paper_style",
        4:"book_style",
        5:"official",
        6:"powerpoint_slides",
        7:"other_slides",
        8:"course_material",
        9:"course_material_scanned",
        10:"official_style_course_mater",
        11:"handwritten",
        12:"pw_protected",
        13:"unsure"
    }

    legend = "Copyright protected documents have a positive label.\n"+\
        "Not copyright protected documents have a negative label.\n"+\
        "False positives are documents that have falsely been predicted as copyrights!\n"+\
        "False negatives are copyright protected documents which were not found!\n\n"+\
        "Precision: true positives / predicted positives\n"+\
        "Recall: true positives / real positives\n"+\
        "F1: 2* precision*recall/(precision+recall)\n\n"

    if(not(isdir(results_path))):
        os.makedirs(results_path)

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    # evaluate the results considering different thresholds
    test_size = len(labels)/len(results[0])

    for t in thres:
        # category evaluation
        category_eval = np.zeros((13,5))
        # get the average scores
        average_scores = np.zeros((8,))

        false_neg_ids = {
            1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[],
            7:[],
            8:[],
            9:[],
            10:[],
            11:[],
            12:[],
            13:[]
        }

        # create a tables
        category_eval_table = PrettyTable(["category","count","false_negative", "false_positive", "not_important","maybe_wrong"])
        results_table = PrettyTable(score_names)
        for k in range(0,len(results[0])):
            indices = results[:,k]>=0
            probs = results[indices,k]
            preds = probs>t
            target = labels[indices]
            test_files = list(compress(doc_ids,indices))

            results_row = []
            results_row.append(accuracy_score(target, preds))
            results_row.append(precision_score(target, preds, average="binary"))
            results_row.append(recall_score(target, preds, average="binary"))
            results_row.append(f1_score(target, preds, average="binary"))
            results_row.extend(confusion_matrix(target, preds).ravel())

            average_scores += np.array(results_row)
            results_row[0:4] = ["%.3f"%(val,) for val in results_row[0:4]]
            results_row[4:] = ["%d   %.3f"%(val,val/test_size) for val in results_row[4:]]
            results_table.add_row(results_row)

            category_eval = update_cat_eval(category_eval, false_neg_ids, test_files, preds)

        results_table.add_row(['']*len(score_names))
        average_scores /= len(results[0])
        average_scores = list(average_scores)
        average_scores[0:4] = ["%.3f"%(val,) for val in average_scores[0:4]]
        average_scores[4:] = ["%d   %.3f"%(int(val),val/test_size) for val in average_scores[4:]]
        results_table.add_row(average_scores)

        with open(join(results_path, filename+"_pred_eval_thres_%.2f.txt"%(t,)), 'w') as f:
            f.write(("Crossvalidation results for threshold %.2f"%(t,)).center(80) + "\n")
            f.write("\n"+legend+"\n\n")
            f.write(results_table.get_string())

        for i in range(len(category_eval)):
            row = [cat_id_to_des[i+1]]
            row.extend(category_eval[i,:])
            row[2:] = ["%d   %.3f"%(int(val),val/row[1]) for val in row[2:]]
            category_eval_table.add_row(row)
        a = ["total"]
        a.extend(np.sum(category_eval, axis=0))
        a[2:] = ["%d   %.3f"%(int(val),val/row[1]) for val in a[2:]]
        category_eval_table.add_row(a)

        with open(join(results_path, filename+"_category_eval_thres_%.2f.txt"%(t,)),"w") as cat_f:
            cat_f.write(("Crossvalidation results").center(80)+"\n")
            cat_f.write(("for different document categories").center(80)+"\n")
            cat_f.write(("for threshold %.2f"%(t,)).center(80) + "\n")
            cat_f.write("\n"+legend+"\n\n")
            cat_f.write(category_eval_table.get_string())
            cat_f.write("\n\n\n")
            cat_f.write("False negative Ids:\n\n")
            for key,vals in false_neg_ids.items():
                cat_f.write(cat_id_to_des[key]+"\n")
                for d_id_t in vals:
                    if(d_id_t[1]==1):
                        cat_f.write(d_id_t[0] + "\txml_error\n")
                    else:
                        cat_f.write(d_id_t[0] + "\n")
                cat_f.write("\n")


### Correlation Investigation ###
def pearson_correlation(data, labels, column_names, results_path):
    from scipy.stats import pearsonr
    res = []
    for i in range(len(data[0])):
        res.append(pearsonr(data[:,i], labels)[0])

    sorted_coefs = sorted(zip(column_names,np.abs(res),res),key=lambda idx: idx[1],reverse=True)
    with open(results_path, "w") as r_f:
        for r in sorted_coefs:
            r_f.write(r[0].ljust(40) + "%.5f"%r[2] + "\n")

def analyse_feature_ranking(classifier, feature_names, max_feat=None, results_path=None):
    params = classifier.get_feature_rating()
    sorted_coefs = sorted(zip(feature_names,np.abs(params),params),key=lambda idx: idx[1],reverse=True)

    if(max_feat is None):
        max_feat = len(feature_names)

    feat_rating = ""
    for r in sorted_coefs[:max_feat]:
        feat_rating += (r[0].ljust(40) + "%.5f"%r[2] + "\n")

    if(not(results_path is None)):
        with open(results_path, "w") as r_f:
            r_f.write(feat_rating)

    return feat_rating


### PREPROCESSING ###
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

def minmax_scaling(features):
    '''
    Normalize the feature matrix. Make sure to handle np.nans beforehand

    @param  features: The feature matrix
    @type   features: np.array

    @return  norm_features: The normalized matrix
    @rtype   norm_features: np.array
    '''
    len_feat = len(features[0])
    max_nor=np.amax(features, axis=0)
    min_nor=np.amin(features, axis=0)
    norm_features = np.zeros(features.shape)
    for i in range(0, len_feat):
        f_range = (max_nor[i]-min_nor[i])
        if(f_range>0):
            norm_features[:,i] = (features[:,i]-min_nor[i])/f_range
        else:
            print("The feature at position %s has always the same value!"%(i,))
    return norm_features

def log_transform(features):
    '''
    Transforms the feature matrix. Make sure to handle np.nans beforehand

    @param  features: The feature matrix
    @type   features: np.array

    @return  trans_features: The transformed matrix
    @rtype   trans_features: np.array
    '''
    trans_features = np.log(features + (-np.min(features,axis=0)+1))
    return trans_features

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

### VISUALIZATION ###
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
    colors = ["red","black"]
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

def plot_curve(x,y,x_label,y_label,curve_label,title):
    plt.figure()
    plt.plot(x, y, color='darkorange',
             lw=2, label=curve_label)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="lower right")
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round_(cm,3)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    args = sys.argv

    train_csv = args[1]
    # test_csv = args[2]

    # train_data, train_labels, train_docs, column_names = load_data(train_csv)
    # test_data, test_labels, test_docs, column_names = load_data(test_csv)

    train = pd.read_csv(train_csv, delimiter=',', header=0, quoting=1)
    train.columns = ["document_id", "class"]
    doc_ids = list(train["document_id"])
    labels = np.array(list(train["class"]))

    # # Create a Logistic Regression Instance
    # from Logistic_Regression import Logistic_Regression
    # kwargs = {"penalty":'l2', "C":1, "fit_intercept":True, "intercept_scaling":1000}
    # lr = Logistic_Regression(**kwargs)
    # results_path = join("/home/kai/Workspace/deep_doc_class/deep_doc_class/results", splitext(basename(train_csv))[0],"new_log_reg")

    # # Do Crossvalidation
    # train_res,test_res = manual_crossvalidation_bin(lr,doc_ids,labels, kfolds=10, shuffle=True, seed=7)
    # analyse_crossvalidation_results_doc_categories(results=test_res, labels=labels, doc_ids=doc_ids, results_path=results_path, filename="test", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)
    # analyse_crossvalidation_results_bin(results=train_res, labels=labels, results_path=results_path, filename="train", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)

    # pearson_correlation(data=train_data, labels=train_labels, column_names=column_names, results_path=join(results_path,"pearson_corr.txt"))
    # analyse_feature_ranking(classifier=lr, feature_names=column_names, results_path=join(results_path,"model_feature_ranking.txt"))

    # Create a Random Forest Instace
    from Random_Forest import Random_Forest
    kwargs = {"n_estimators":1000, "criterion":'gini', "max_depth":15, "min_samples_split":2, "min_samples_leaf":1, "min_weight_fraction_leaf":0.0, "max_features":'auto', "max_leaf_nodes":None, "min_impurity_split":1e-07, "bootstrap":True, "oob_score":False, "n_jobs":1, "random_state":None, "verbose":0, "warm_start":False, "class_weight":None}
    rf = Random_Forest(**kwargs)
    results_path = join("/home/kai/Workspace/deep_doc_class/deep_doc_class/results", splitext(basename(train_csv))[0],"folds5_md15_e1k_random_forest")

    if(not(isdir(results_path))):
        os.makedirs(results_path)

    # Do Crossvalidation
    train_res,test_res = manual_crossvalidation_bin(rf,doc_ids,labels, results_path, kfolds=5, shuffle=True, seed=7)
    analyse_crossvalidation_results_doc_categories(results=test_res, labels=labels, doc_ids=doc_ids, results_path=results_path, filename="test", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)
    analyse_crossvalidation_results_bin(results=train_res, labels=labels, results_path=results_path, filename="train", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)


    # # Create a MLP Instance
    # from Keras_Dense_MLP import Keras_Dense_MLP
    # layer_params = {"kernel_initializer":"glorot_uniform", "activation":"sigmoid"}
    # compile_params = {"loss":"mean_squared_error", "optimizer":"sgd", "metrics":["accuracy"]}
    # train_params = {"epochs":100000, "batch_size":64, "verbose":0}
    # mlp = Keras_Dense_MLP(neuron_layers=[len(train_data[0]),500,1], layer_params=layer_params, compile_params=compile_params, **train_params)

    # # Do Crossvalidation
    # train_res,test_res = crossvalidate_proba_bin(lr,data=train_data, labels=train_labels, kfolds=10, shuffle=True, seed=7)
    # analyse_crossvalidation_results_doc_categories(results=test_res, labels=train_labels, doc_ids=train_docs, results_path=results_path, filename="test", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)
    #
    # analyse_crossvalidation_results_bin(results=train_res, labels=train_labels, results_path=results_path, filename="train", thres=[0.5,0.4,0.3,0.2,0.1], boxplots=False)

    # # Compare parameter combinations
    # results_file = "MLP_evaluation.txt"
    # comb_params = {"neuron_layers":[[len(train_data[0]),64,32,1],[len(train_data[0]),64,1],[len(train_data[0]),64,32,16,1]]}
    # compare_parameter_combinations_bin(mlp, comb_params, train_data, train_labels, test_data, test_labels, results_file, True)
