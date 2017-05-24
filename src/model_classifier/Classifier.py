# Classifier.py
import csv
import numpy as np

class Classifier(object):

    def __init__(self,**kwargs):
        '''
        Initializes a classifier using all the parameters specified in kwargs. Since in most cases this will only be a wrapper around a sklearn classifier there should be a variable self.model which references the model_class, whenever possible. Since a model might need to be re_initilize the parameters should be stored as they are.

        It might be sensible and way more convenient to have more input arguments. As long as this does not require a change in any other function's signiture that is okay.

        Especially the functions re-initilize and get_attribute and set_attribute might get more complicated when using more input arguments.

        @param  **kwargs: A dictionary containing all the parameters need to initialize the model.
        @type   **kwargs: dict
        '''
        raise NotImplementedError('users must define __init__() to use this base class')

    def re_initialize(self):
        '''
        Re-initializes the model of the classifier stored in the instance variable self.model using the stored parameters.
        '''
        raise NotImplementedError('users must define initialize() to use this base class')

    def get_attribute(self, param_key):
        '''
        Returns the value of the parameter referenced by the attribute param_key.

        @param  param_key: The name of a parameter of the model
        @type   param_key: str

        @return  param_val: The value of the parameter
        @rtype   param_val: any
        '''
        raise NotImplementedError('users must define get_attribute() to use this base class')

    def set_attribute(self, param_key, param_val):
        '''
        Replaces the value of the parameter referenced by the attribute param_key with the new value param_val.

        @param  param_key: The name of a parameter of the model
        @type   param_key: str

        @param  param_val: The new value for the parameter
        @type   param_val: any
        '''
        raise NotImplementedError('users must define set_attribute() to use this base class')

    def save_model(self, abs_filepath):
        '''
        Serializes the current state of the model and writes it to the file specified by the abs_filepath.

        @param  abs_filepath: The absolute filepath to the file where the model should be stored
        @type   abs_filepath: str
        '''
        raise NotImplementedError('users must define save_model() to use this base class')

    def load_model(self, abs_filepath):
        '''
        Loads a model from the filepath specified abs_filepath and derializes it into the instance variable model.

        @param  abs_filepath: The absolute filepath where the model is stored
        @type   abs_filepath: str
        '''
        raise NotImplementedError('users must define load_model() to use this base class')

    def train(self, data, labels):
        '''
        Trains (or fits according to the naming in the sklearn library) the model using the provided data and labels. The input argument data should be a numeric 2-dimensional numpy array with the datapoints as row vectors and feature columns. The labels should be a numeric 1-dimensional numpy array with the same row count as the first dimension of the data matrix. Rows with a similar index in data and labels belong together.

        @param  data: The data as row vectors (column features).
        @type   data: np.array((n,m))

        @param  labels: The labels as one dimensional vector.
        @type   labels: np.array((n,))
        '''
        raise NotImplementedError('users must define fit() to use this base class')

    def eval_accuracy(self, data):
        '''
        Evaluates the accuracy of a trained model on the given new data. The data should be in the same format as in the train case.

        @param  data: The data which is to be evaluated
        @type   data: np.array((n,m))

        @return  acc: The accuracy of the prediction.
        @rtype   acc: float
        '''
        raise NotImplementedError('users must define evaluate() to use this base class')

    # Im not entirely sure yet if those last 4 functions should be reduced to 2
    def predict_proba_nary(self, data):
        '''
        Computes the prediction probabilities of the trained model for the given data. The data should be in the same format as in the train case. The probabilities should be the probability for all different classes (a 2-dimensional numpy array) with one column for each label. The values in a row should add up to one.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  proba: The prediction probabilities for the input data.
        @rtype   proba: np.array((n,c))
        '''
        raise NotImplementedError('users must define predict_proba() to use this base class')

    def predict_proba_binary(self, data):
        '''
        Computes the prediction probabilities of the trained model for the given data. The data should be in the same format as in the train case. The probabilities should be the probability for class 1 (1-dimensional np.array). Since the probability of the class 0 is just the complement that adds up to one it can be omitted.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  proba: The prediction probabilities for the input data.
        @rtype   proba: np.array((n,c))
        '''
        raise NotImplementedError('users must define predict_proba() to use this base class')

    def predict_nary(self, data):
        '''
        Computes the prediction of the trained model for the given data. The data should be in the same format as in the train case. The prediction should be a 1-dimensional numpy array with the same row count as the input data and the same range of values as there have been labels in the train.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  pred: The prediction for each data vector.
        @rtype   pred: bin:np.array((n,)) n-ary:np.array((n,c))
        '''
        raise NotImplementedError('users must define predict_mult() to use this base class')

    def predict_binary(self, data, thres=0.5):
        '''
        Does the same as predict_mult but for binary classifications only. There is an extra function for this scenario because there is an easy way to adjust the classification since the prediction probabilities in a binary case are counterparts where the probability for one can be inferred from the other class.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  pred: The prediction for each data vector.
        @rtype   pred: np.array((n,))
        '''
        raise NotImplementedError('users must define predict_bin() to use this base class')
