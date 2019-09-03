# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:09:40 2017
^O@author: odrec
"""
import os, json
from os.path import join, isdir, basename
import numpy as np
import features_names
import logging.config
import keras
from keras.optimizers import Nadam
from keras.layers import Input, Dense, Dropout
from keras.models import model_from_json
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import ensemble
import joblib
import paths
logging.config.fileConfig(fname=paths.LOG_FILE, disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")
def predict(features, model):
    '''
    Makes the predictions for the provided features

    @param features: numpy array with the features
    @dtype features: numpy array
    @param model: prediction model
    @dtype model: keras model

    @return positive_prob: prediction
    @rtype positive_prob: numpy array
    '''
    features = np.nan_to_num(features) #this shouldn't be needed if normalization is correct
    prd = model.predict_proba(features)
    positive_prob = []
    for i,p in enumerate(prd):
        positive_prob.append(prd[i][1])
    return positive_prob

def load_trained_model(model_path, model_name):
    '''
    Loads the trained model
    @param model_path: path to the model
    @dtype model_path: str

     @return model: loaded model
    @rtype model: keras model
    '''
    model = None
    if isdir(model_path):
        model_file = join(model_path, model_name)
        with open(model_file+'.json','r') as f:
            model_json = json.load(f)
        model = model_from_json(model_json)
        model.load_weights(model_file+'.h5')
    else: model = None
    return model

def create_model(feature_name):
    '''
    Creates a model depending on the feature.
    @param feature_name: name of the feature
    @dtype feature_name: str

    @return model: created model
    @rtype model: sklearn model (the type of model depends on the feature)
    '''
    if feature_name == 'filename':
        model = linear_model.SGDClassifier(penalty="elasticnet", loss="modified_huber", \
                                                        tol=1e-2, class_weight="balanced")
    elif feature_name == 'folder_name': model = naive_bayes.MultinomialNB(alpha=1.0)
    elif feature_name == 'creator': model = naive_bayes.MultinomialNB(alpha=1.0)
    elif feature_name == 'producer': model = naive_bayes.MultinomialNB(alpha=1.0)
    elif feature_name == 'numeric': model = ensemble.RandomForestClassifier(n_estimators=1000, criterion="entropy",\
                                                                            max_features="auto", class_weight="balanced",\
                                                                            min_samples_split=10, max_depth=None)
    else:
        #The rest of the text features enter here.
        model = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
#        model = ensemble.RandomForestClassifier(n_estimators=100, criterion="gini", max_features="sqrt",\
#                                                        class_weight="balanced", min_samples_split=10)
    return model

def create_final_model():
    model = ensemble.RandomForestClassifier(n_estimators=1000)
    return model

def train_final_model(train_features_data, train_labels_data, models_path, meta=False, deep=False):
    '''
    Trains the final model.
    @param train_features_data: array with the training data
    @dtype train_features_data: array

    @param train_labels_data: list with the training labels
    @dtype train_labels_data: list

    @param models_path: path to save the model
    @dtype models_path: str

    @param meta: used as flag to see if metadata was provided
    @dtype meta: bool

    @param deep: flag to see if there are deep features
    @dtype deep: bool

    @return final_model: trained model
    @rtype final_model: sklearn model

    @return scores: scores for the data (only when training)
    @rtype scores: array
    '''
    if not isdir(models_path): os.makedirs(models_path)
    model_name = "final"
    if meta:
        model_name = model_name + "_meta"
    if deep:
        model_name = model_name + "_deep"
    model_name = model_name + ".pkl"
    final_model = create_final_model()
    final_model = fit_model(final_model, train_features_data, train_labels_data)
    save_model(final_model, models_path, model_name)
    return final_model

def get_final_model_output(features_data, final_model=None, labels_data=None, models_path="", meta=False, deep=False):
    '''
    Gets the output scores of the final model.

    @param model: trained model
    @dtype model: sklearn model
    @param features_data: array with the data
    @dtype features_data: array

    @param labels_data: list with the labels in case of testing
    @dtype labels_data: list

    @param models_path: path to load the model
    @dtype models_path: str

    @param meta: used as flag to see if model uses metadata
    @dtype meta: bool

    @param deep: flag to see if model uses deep features
    @dtype deep: bool

    @return scores: score for the model
    @rtype scores: array

    @return final_predictions: list of predictions for the data
    @rtype final_predictions: list
    '''
    final_scores = None
    final_predictions = None
    model_name = ""
    if not final_model:
        model_name = model_name + "final"
        if meta:
            model_name = model_name + "_meta"
        if deep:
            model_name = model_name + "_deep"
        model_name = model_name + ".pkl"
        final_model = load_sklearn_model(model_name, None, models_path)
    if labels_data:
        final_scores = score_model(final_model, features_data, labels_data)
    else:
        debuglogger.debug("Predicting outputs with final classifier.")
        final_predictions = predict(features_data, final_model)
        debuglogger.debug("Finishing predicting outputs for final classifier.")
    return final_scores, final_predictions

def create_deep_model(X_train):
    '''
    Creates a small neural network model for the output of the deep models.
    @param X_train: data for training used to generate the size of the input layer
    @dtype X_train: array

    @return model: created model
    @rtype model: keras model
    '''
    input_tensor = Input(X_train.shape[1:])
    x = input_tensor
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(input_tensor, x)
    model.compile(optimizer=Nadam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def fit_deep_model(model, X_train, y_train):
    '''
    Fits the neural network model for the deep networks to the training data.

    @param model: untrained model
    @dtype model: keras model
    @param X_train: training data
    @dtype X_train: array

    @param y_train: labels for the training data
    @dtype y_train: list

    @return model: trained model
    @rtype model: keras model
    '''
    batch_size = 1
    model.fit(X_train, y_train, batch_size=batch_size*4, epochs=10)
    model.fit(X_train, y_train, batch_size=batch_size*4, epochs=10)
    model.optimizer.lr = 1e-5
    model.fit(X_train, y_train, batch_size=batch_size*4, epochs=10)
    model.optimizer.lr = 1e-6
    model.fit(X_train, y_train, batch_size=batch_size*4, epochs=10)
    model.optimizer.lr = 1e-9
    model.fit(X_train, y_train, batch_size=batch_size*4, epochs=20)
    return model

def pseudo_labeling(model, y_pseudo, X_train, y_train, U_train):
    i_trn = 0
    i_test = 0
    y_pseudo = y_pseudo.flatten()
    #y_train = np.transpose(y_train,(1,0))

    # iterate through 800 mini-batch
    num_iter = 1100*2
    # mini-batch size
    size_trn = 24
    size_test = 8
    num_batch_per_epoch_trn = int(X_train.shape[0]/size_trn)
    if num_batch_per_epoch_trn == 0: num_batch_per_epoch_trn == 1
    num_batch_per_epoch_test = int(U_train.shape[0]/size_test)
    if num_batch_per_epoch_test == 0: num_batch_per_epoch_test == 1
    index_trn = np.random.permutation(num_batch_per_epoch_trn)
    index_test = np.random.permutation(num_batch_per_epoch_test)
    for i in range(num_iter):
        if num_batch_per_epoch_trn == 0: i_trn
        else: i_trn = index_trn[i%num_batch_per_epoch_trn]

        if num_batch_per_epoch_trn == 0: i_trn
        else: i_test = index_test[i%num_batch_per_epoch_test]
        comb_features = np.concatenate((X_train[(size_trn*i_trn):size_trn*(i_trn+1)],
                                       U_train[(size_test*i_test):size_test*(i_test+1)]),axis=0)
        comb_labels = np.concatenate((y_train[(size_trn*i_trn):size_trn*(i_trn+1)],
                                     y_pseudo[(size_test*i_test):size_test*(i_test+1)]), axis=0)
        res = model.train_on_batch(comb_features, comb_labels)
#        print(res,model.metrics_names)
        if num_batch_per_epoch_trn > 0 and (i+1)%num_batch_per_epoch_trn == 0:
            index_trn = np.random.permutation(num_batch_per_epoch_trn)
        else: index_trn = np.random.permutation(1)
        if num_batch_per_epoch_test > 0 and (i+1)%num_batch_per_epoch_test == 0:
            index_test = np.random.permutation(num_batch_per_epoch_test)
        else: index_test = np.random.permutation(1)
    return model

def train_deep_model(models_path, X_train, y_train, U_train, unlabeled_flag):
    '''
    Trains the model.

    @param models_path: path to save the models
    @dtype models_path: str
    @param X_train: array with the training data
    @dtype X_train: array

    @param y_train: list with the training labels
    @dtype y_train: list

    @param U_train: array with the unlabeled data (for pseudo-labeling)
    @dtype U_train: array

    @param unlabeled_flag: flag to whether to use unlabeled data
     @dtype unlabeled_flag: bool

    @return predictions: predictions for the data (only when not training)
    @rtype predictions: array

    @return scores: scores for the data (only when training)
    @rtype scores: array
    '''
    logger.debug("Training deep classifier.")
    debuglogger.debug("Training deep classifier.")
    batch_size = 1
    deep_model = create_deep_model(X_train)
    try:
        deep_model.load_weights(join(models_path,'deep_model.h5'))
        logger.debug("Trained deep classifier loaded.")
        debuglogger.debug("Trained deep classifier loaded.")
    except:
        logger.debug("Trained deep classifier created.")
        debuglogger.debug("Trained deep classifier created.")
    deep_model = fit_deep_model(deep_model, X_train, y_train)
    if unlabeled_flag:
        logger.debug("Doing pseudo-labelling training.")
        debuglogger.debug("Doing pseudo-labelling training.")
        y_pseudo = deep_model.predict(U_train, batch_size=batch_size)
        deep_model = pseudo_labeling(deep_model, y_pseudo, X_train, y_train, U_train)
    if not isdir(models_path): os.makedirs(models_path)
    model_file = join(models_path,'deep_model')
    model_json = deep_model.to_json()
    with open(model_file+".json", "w+") as json_file:
        json.dump(model_json, json_file)
    deep_model.save_weights(model_file+'.h5')
    logger.debug("Deep classifier saved on %s."%model_file)
    debuglogger.debug("Deep classifier saved on %s."%model_file)
    debuglogger.debug("Training deep classifier.")
    logger.debug("Finished training deep classifier.")
    debuglogger.debug("Finished training deep classifier.")
    return deep_model

def get_deep_model_output(features_matrix, deep_model=None, models_path="", matrix_ids=None, doc_ids=None):
    '''
    Gets the deep model's outputs.
    @param features_matrix: matrix with the data for prediction
    @dtype features_matrix: array

    @param deep_model: trained deep model
    @dtype deep_model: keras model

    @param models_path: path to load the models
    @dtype models_path: str

    @param matrix_ids: list of ids to having the order of the matrix data (per page)
    @dtype matrix_ids: list

    @param doc_ids: list of doc ids to arrange shuffled data if required
    @dtype doc_ids: list

    @return average_predictions: prediction probabilities averaged for the pages from each document
    @rtype average_predictions: list
    '''
    if not deep_model:
        if isdir(models_path):
            deep_model_name = 'deep_model'
            deep_model = load_trained_model(models_path, deep_model_name)
    features_matrix=np.array([np.array(xi) for xi in features_matrix])
    if features_matrix.shape[0] == 3: features_matrix = np.concatenate(features_matrix, axis=1)
    pred = deep_model.predict(features_matrix)
    average_predictions = []
    sum_values = 0
    num_pages = 0
    new_ids = []
    for did in doc_ids:
        sum_values = 0
        num_pages = 0
        for i,mid in enumerate(matrix_ids):
            if mid == did:
                sum_values += pred[i]
                num_pages += 1
                new_ids.append(matrix_ids[i])
            if i == len(matrix_ids)-1:
                if num_pages > 0: 
                    average_predictions.append(sum_values/num_pages)
                else:
                    average_predictions.append(sum_values)
                    logger.warning("NO PAGES for file %s"%did)
    return average_predictions

def fit_model(model, data, labels):
    '''
    Fits a model to the training data.

    @param model: untrained model
     @dtype model: sklearn model
    @param data: training data
    @dtype data: array

    @param labels: labels for the training data
    @dtype labels: list

    @return model: trained model
     @rtype model: sklearn model
    '''
    model.fit(data, labels)
    return model

def score_model(model, data, labels):
    '''
    Calculates the score of a model with the testing data.

    @param model: trained model
    @dtype model: sklearn model
    @param data: testing data
    @dtype data: array

    @param labels: labels for the testing data
    @dtype labels: list

    @return score: score for the testing of the model
    @rtype score: array
    '''
    score = model.score(data, labels)
    return score

def save_model(model, model_path, model_name=None, feature_name=None):
    '''
    Saves the trained model to a .pkl file.

     @param model: trained model
     @dtype model: sklearn model

    @param model_path: path to where the model will be saved
    @dtype model_path: str
    @param model_name: name of the model
     @dtype model_name: str/None

    @param feature_name: name of the feature
    @dtype feature_name: str/None
    '''
    if not model_name and feature_name:
        model_name = join(model_path, feature_name+'_model.pkl')

    debuglogger.debug("Saving trained model for model %s."%(basename(model_name)))
    model_path = join(model_path, model_name)
    joblib.dump(model, model_path)
    debuglogger.debug("Trained model for model %s succesfully saved on %s."%(basename(model_name),model_path))

def load_sklearn_model(model_name=None, feature_name=None, model_path=""):
    '''
    Loads the trained model from a .pkl file.
    @param model_name: name of the model
     @dtype model_name: str/None

    @param feature_name: name of the feature
    @dtype feature_name: str/None

    @param model_path: path of the model
    @dtype model_path: str

    @return model: the loaded model
     @rtype model: sklearn model
    '''
    if not model_name:
        model_path = join(model_path ,feature_name+'_model.pkl')

    else:
        model_path = join(model_path ,model_name)
    model = joblib.load(model_path)
    return model

def train_structural_models(train_features_data, train_labels_data, \
                            test_features_data, test_labels_data, models_path, meta=False):
    '''
    Trains the models.
    @param train_features_data: array with the training data
    @dtype train_features_data: array

    @param train_labels_data: list with the training labels
    @dtype train_labels_data: list

    @param test_features_data: array with the testing data (only used when training)
    @dtype test_features_data: array

    @param train_labels_data: list of labels for the training data (only used when training)
    @dtype train_labels_data: list

    @param test_labels_data: list of labels for the testing data (only used when training)
    @dtype test_labels_data: list

    @param models_path: path to save the models
    @dtype models_path: str

    @param meta: used as flag to see if metadata was provided
    @dtype meta: bool

    @return predictions: predictions for the data (only when not training)
    @rtype predictions: array

    @return scores: scores for the data (only when training)
    @rtype scores: array
    '''
    if not isdir(models_path): os.makedirs(models_path)
    if meta:
        bow_features_names = features_names.bow_text_features +\
        features_names.bow_prop_features + features_names.bow_meta_features

    else:
        bow_features_names = features_names.bow_text_features + features_names.bow_prop_features
    #Scores for all BOW features plus numeric features
    scores = [None] * (len(bow_features_names)+1)
    models = [None] * (len(bow_features_names)+1)
    for b,bf in enumerate(bow_features_names):
        debuglogger.debug("Training classifier for %s feature."%(bf))
        models[b] = create_model(bf)
        models[b] = fit_model(models[b], train_features_data[b], train_labels_data)
        debuglogger.debug("Finished training classifier for %s feature."%(bf))
        debuglogger.debug("Testing classifier for %s feature."%(bf))
        scores[b] = score_model(models[b], test_features_data[b], test_labels_data)
        debuglogger.debug("Finished testing classifier for %s feature. Score: %s."%(bf,str(scores[b])))
        save_model(models[b], models_path, None, bf)
        logger.info("Classifier for %s feature had a score of %s while testing."%(bf,str(scores[b])))
    #Train structural (numeric) features model
    models[-1] = create_model('numeric')
    debuglogger.debug("Training classifier for numeric features.")
    models[-1] = fit_model(models[-1], train_features_data[-1], train_labels_data)
    debuglogger.debug("Finished training classifier for numeric features.")
    debuglogger.debug("Testing classifier for numeric features.")
    scores[-1] = score_model(models[-1], test_features_data[-1], test_labels_data)
    debuglogger.debug("Finished testing classifier for numeric features. Score: %s."%(str(scores[-1])))
    debuglogger.debug("Saving trained classifier for numeric feature.")
    save_model(models[-1], models_path, None, 'numeric')
    logger.info("Classifier for numeric feature had a score of %s while testing."%(str(scores[-1])))
    return models, scores

def get_models_output(features_data, models=[], models_path="", meta=False):
    '''
    Gets the models' outputs.
    @param features_data: array with the data for prediction
     @dtype features_data: array

    @param models: list of trained models
    @dtype models: list of model objects

    @param meta: used as flag to see if metadata was provided
    @dtype meta: list/bool

    @param models_path: path to load the models
    @dtype models_path: str

    @param meta: used as flag to see if metadata was provided
    @dtype meta: bool

    @return predictions_prob: prediction probabilities for the data
    @rtype predictions_prob: list

    @return positive_prob: the positive probability for the data
    @rtype positive_prob: list
    '''
    if meta:
        bow_features_names = features_names.bow_text_features +\
        features_names.bow_prop_features + features_names.bow_meta_features

    else:
        bow_features_names = features_names.bow_text_features + features_names.bow_prop_features
    #Predictions for all BOW features plus numeric features
    predictions_prob = [None] * (len(bow_features_names)+1)
    positive_prob = [None] * (len(bow_features_names)+1)
    for b,bf in enumerate(bow_features_names):
        positive_prob[b] = [None] * len(features_data[b])
        debuglogger.debug("Predicting outputs with classifier for %s feature."%(bf))
        logger.info("Predicting outputs with classifier for %s feature."%(bf))
        if models: model = models[b]
        else: model = load_sklearn_model(None, bf, models_path)
        predictions_prob[b] = model.predict_proba(features_data[b])
        for i,v in enumerate(predictions_prob[b]):
            positive_prob[b][i] = predictions_prob[b][i][1]
        debuglogger.debug("Finishing predicting outputs with classifier for %s feature."%(bf))
        logger.info("Finishing predicting outputs with classifier for %s feature."%(bf))
    debuglogger.debug("Predicting outputs with classifier for numeric feature.")
    logger.info("Predicting outputs with classifier for numeric feature.")
    if models: model = models[-1]
    else: model = load_sklearn_model(None, 'numeric', models_path)
    features_data[-1] = np.nan_to_num(features_data[-1]) #this shouldn't be needed if normalization is correct
    predictions_prob[-1] = model.predict_proba(features_data[-1])
    positive_prob[-1] = [None] * len(features_data[-1])
    for i,v in enumerate(predictions_prob[b]):
        positive_prob[-1][i] = predictions_prob[-1][i][1]
    debuglogger.debug("Finishing predicting outputs with classifier for numeric feature.")
    logger.info("Finishing predicting outputs with classifier for numeric feature.")
    return predictions_prob, positive_prob