# Keras_Dense_MLP
from Classifier import Classifier
import numpy as np

# https://keras.io/
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import metrics

class Keras_Dense_MLP(Classifier):

    compile_keys = ["optimizer", "loss", "metrics", "loss_weights", "sample_weight_mode"]
    layer_keys = ["activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint"]

    def __init__(self, neuron_layers, layer_params, compile_params, **train_params):
        '''
        Initializes a multi-layer-perceptron using the keras api. The multiple layer-sizes have to be specified in neuron layers, where the first entry has to be the input dimensions and the last one the output dimensions. This model will only allow 1-dimensional input. Layer_params should contain all the paramaters that the class keras.layers.Dense can hanlde like activation function and so on. All those parameters are defined for all layers simultaneously. Compile_params contains all the parameters needed to compile a model.

        @param  neuron_layers: A list containing all sizes of each layer (eg. [10,100,1] would be a network with 10 input neurons, one hidden layer with 100 neurons and an output layer of only one neuron)
        @type   neuron_layers: list

        @param  **layer_params: A dictionary containing all the parameters for initializing a layer.
        @type   **layer_params: dict

        @param  **compile_params: A dictionary containing all the parameters for compiling a model.
        @type   **compile_params: dict
        '''
        self.neuron_layers = neuron_layers
        self.layer_params = layer_params
        self.compile_params = compile_params
        self.train_params = train_params

        if(not("optimizer" in compile_params)):
            print("Keras requires the parameter optimizer to be specified but it is not in compile_params! Using sgd as optimizer!")
            self.compile_params["optimizer"] = "sgd"
        if(not("loss" in compile_params)):
            print("Keras requires the parameter loss to be specified but it is are not in compile_params! Using mean_squared_error as loss function!")
            self.compile_params["loss"] = "mean_squared_error"
        compile_params["metrics"] = ["accuracy"]

        self.model = Sequential()

        # Create the first layer
        self.model.add(Dense(self.neuron_layers[1], input_shape=(self.neuron_layers[0],), **self.layer_params))

        # Create all other layers
        for i in range(2,len(self.neuron_layers)):
            self.model.add(Dense(self.neuron_layers[i], **self.layer_params))

        # Compile model
        self.model.compile(**self.compile_params)

    def re_initialize(self):
        '''
        Re-initializes the model using the parameters stored during the first initialization.
        '''
        self.model = Sequential()

        # Create the first layer
        self.model.add(Dense(self.neuron_layers[1], input_shape=(self.neuron_layers[0],), **self.layer_params))

        # Create all other layers
        for i in range(2,len(self.neuron_layers)):
            self.model.add(Dense(self.neuron_layers[i], **self.layer_params))

        # Compile model
        self.model.compile(**self.compile_params)

    def get_attribute(self, param_key):
        '''
        Returns the value of the parameter referenced by the attribute param_key.

        @param  param_key: The name of a parameter of the model
        @type   param_key: str

        @return  param_val: The value of the parameter
        @rtype   param_val: any
        '''
        if(param_key=="neuron_layers"):
            return self.neuron_layers
        elif(param_key in layer_params):
            return self.layer_params[param_key]
        elif(param_key in compile_params):
            return self.compile_params[param_key]
        else:
            print("This key does not exist!")
            return None

    def set_attribute(self, param_key, param_val):
        '''
        Replaces the value of the parameter referenced by the attribute param_key with the new value param_val.

        @param  param_key: The name of a parameter of the model
        @type   param_key: str

        @param  param_val: The new value for the parameter
        @type   param_val: any
        '''
        if(param_key=="neuron_layers"):
            self.neuron_layers = param_val
        elif(param_key in KerasDenseMLP.layer_keys):
            self.layer_params[param_key] = param_val
        elif(param_key in KerasDenseMLP.compile_keys):
            self.compile_params[param_key] = param_val
        else:
            print("This is not a valid key for this KerasDenseMLP!")
            return None

    def save_model(self, abs_filepath):
        '''
        Serializes the current state of the model and writes it to the file specified by the abs_filepath.

        @param  abs_filepath: The absolute filepath to the file where the model should be stored
        @type   abs_filepath: str
        '''
        model.save(abs_filepath)

    def load_model(self, abs_filepath):
        '''
        Loads a model from the filepath specified abs_filepath and derializes it into the instance variable model.

        @param  abs_filepath: The absolute filepath where the model is stored
        @type   abs_filepath: str
        '''
        self.model = load_model(abs_filepath)

    def train(self, data, labels):
        '''
        Trains (or fits according to the naming in the sklearn library) the model using the provided data and labels. The attribute data should be a numeric 2-dimensional numpy array data with the datapoints as row vectors and the features columns. The labels should be a numeric 1-dimensional numpy array with the same row count as the first dimension of the data matrix. Rows with a similar index in data and labels belong together.

        @param  data: The data as row vectors (column features).
        @type   data: np.array((n,m))

        @param  labels: The labels as one dimensional vector.
        @type   labels: np.array((n,))
        '''
        self.model.fit(x=data, y=labels, **self.train_params)

    def eval_accuracy(self, data):
        '''
        Evaluates the accuracy of a trained model on the given new data. The data should be in the same format as in the train case.

        @param  data: The data which is to be evaluated
        @type   data: np.array((n,m))

        @return  acc: The accuracy of the prediction.
        @rtype   acc: float
        '''
        return self.model.evaluate(data, labels)[1]*100

    def predict_proba_nary(self, data):
        '''
        Computes the prediction probabilities of the trained model for the given data. The data should be in the same format as in the train case. The probabilities should be the probability for all different classes (a 2-dimensional numpy array) with one column for each label. The values in a row should add up to one.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  proba: The prediction probabilities for the input data.
        @rtype   proba: np.array((n,c))
        '''
        return self.model.predict(data)

    def predict_proba_binary(self, data):
        '''
        Computes the prediction probabilities of the trained model for the given data. The data should be in the same format as in the train case. The probabilities should be the probability for class 1 (1-dimensional np.array). Since the probability of the class 0 is just the complement that adds up to one it can be omitted.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  proba: The prediction probabilities for the input data.
        @rtype   proba: np.array((n,c))
        '''
        if(self.neuron_layers[-1]==1):
            return self.model.predict(data).ravel()
        elif(self.neuron_layers[-1]==2):
            return self.model.predict(data)[:,1].ravel()
        else:
            print("Not a binary classification network!")
            sys.exit(1)

    def predict_nary(self, data):
        '''
        Computes the prediction of the trained model for the given data. The data should be in the same format as in the train case. The prediction should be a 1-dimensional numpy array with the same row count as the input data and the same range of values as there have been labels in the train.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  pred: The prediction for each data vector.
        @rtype   pred: np.array((n,c))
        '''
        return np.argmax(self.model.predict(data), axis=1)

    def predict_binary(self, data, thres=0.5):
        '''
        Does the same as predict_mult but for binary classifications only. There is an extra function for this scenario because there is an easy way to adjust the classification since the prediction probabilities in a binary case are counterparts where the probability for one can be inferred from the other class.

        @param  data: The data which is to be predicted.
        @type   data: np.array((n,m))

        @return  pred: The prediction for each data vector.
        @rtype   pred: np.array((n,))
        '''
        return self.model.predict(data)[:]>thres
