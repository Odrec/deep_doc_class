# Keras_Dense_MLP
from Classifier import Classifier
import numpy as np

# https://keras.io/
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import metrics

class Keras_Dense_MLP():

    compile_keys = ["optimizer", "loss", "metrics", "loss_weights", "sample_weight_mode"]
    layer_keys = ["activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint"]
    train_keys = ["batch_size", "epochs", "verbose", "callbacks", "validation_split", "validation_data", "shuffle", "class_weight", "sample_weight", "initial_epoch"]

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

    def get_params(self):
        '''
        Returns all paramters in a dictionary.

        @return  all_params: The dictionary containing all parameter
        @rtype   param_val: dict
        '''
        all_params = {}
        all_params["neuron_layers"] = self.neuron_layers
        all_params.update(self.layer_params)
        all_params.update(self.compile_params)
        all_params.update(self.train_params)
        return all_params

        if(param_key=="neuron_layers"):
            return self.neuron_layers
        elif(param_key in self.layer_keys):
            return self.layer_params[param_key]
        elif(param_key in self.compile_keys):
            return self.compile_params[param_key]
        elif(param_key in self.train_keys):
            return self.train_params[param_key]
        else:
            print("This key does not exist!")
            return None

    def set_params(self, **params):
        '''
        Replaces parameter values.

        @param  param_key: All parameters that are supposed to be reset
        @type   param_key: dict
        '''
        for param_key, param_val in params.items():
            if(param_key=="neuron_layers"):
                self.neuron_layers = param_val
            elif(param_key in self.layer_keys):
                self.layer_params[param_key] = param_val
            elif(param_key in self.compile_keys):
                self.compile_params[param_key] = param_val
            elif(param_key in self.train_keys):
                self.train_params[param_key] = param_val
            else:
                print("This is not a valid key for this KerasDenseMLP!")
                return None
        self.re_initialize()

    def get_deepcopy(self):
        copy = Keras_Dense_MLP(self.neuron_layers, self.layer_params, self.compile_params, **self.train_params)
        for i,layer in enumerate(self.model.layers):
            copy.model.layers[i].set_weights(layer.get_weights())
        return copy

    def save_model(self, abs_filepath):
        '''
        Serializes the current state of the model and writes it to the file specified by the abs_filepath.

        @param  abs_filepath: The absolute filepath to the file where the model should be stored
        @type   abs_filepath: str
        '''
        self.model.save(abs_filepath)

    def load_model(self, abs_filepath):
        '''
        Loads a model from the filepath specified abs_filepath and derializes it into the instance variable model.

        @param  abs_filepath: The absolute filepath where the model is stored
        @type   abs_filepath: str
        '''
        self.model = load_model(abs_filepath)

    def fit(self, X, y):
        '''
        Trains (or fits according to the naming in the sklearn library) the model using the provided data and labels. The attribute X should be a numeric 2-dimensional numpy array with the datapoints as row vectors and the features columns. The attribut y should be a numeric 1-dimensional numpy array with the same row count as the first dimension of the data matrix signifying the target labels. Rows with a similar index in data and labels belong together.

        @param  X: The data as row vectors (column features).
        @type   X: np.array((n,m))

        @param  y: The labels as one dimensional vector.
        @type   y: np.array((n,))
        '''
        self.model.fit(x=X, y=y, **self.train_params)

    def predict_proba(self, X):
        '''
        Computes the prediction probabilities of the trained model for the given data. The data should be in the same format as in the train case. The probabilities should be the probability for all different classes (a 2-dimensional numpy array) with one column for each label. The values in a row should add up to one.

        @param  X: The data which is to be predicted.
        @type   X: np.array((n,m))

        @return  final_probs: The prediction probabilities for the input data.
        @rtype   final_probs: np.array((n,c))
        '''
        if(self.neuron_layers[-1]==1):
            probs = self.model.predict(X).ravel()
            probs -= np.min(probs)
            max_p = np.max(probs)
            if(max_p>0):
                probs /= max_p
            final_probs = np.zeros((len(probs),2))
            final_probs[:,0] = (-1)*probs + 1
            final_probs[:,1] = probs
            return final_probs
        else:
            probs = self.model.predict(X)
            probs /= np.sum(probs,axis=1)
            return probs

    def predict(self, X):
        '''
        Computes the prediction label of the trained model for the given data. The data should be in the same format as in the train case.

        @param  X: The data which is to be predicted.
        @type   X: np.array((n,m))

        @return  proba: The prediction labels for the input data.
        @rtype   proba: np.array(n)
        '''
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        '''
        Evaluates the accuracy of a trained model on the given new data. The data should be in the same format as in the train case.

        @param  X: The data which is to be evaluated
        @type   X: np.array((n,m))

        @param  y: The target labels
        @type   y: np.array(n)

        @return  acc: The accuracy of the prediction.
        @rtype   acc: float
        '''
        return self.model.evaluate(X, y)[1]*100

if __name__ == "__main__":
    DEEP_DOC_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class"
    DATA_PATH = join(DEEP_DOC_PATH, "data")
    train_file = "train_2017_08_10_cleanup.csv"
    train_data_path = join(DATA_PATH,"feature_values",train_file)
    train_data, train_labels, train_docs, column_names = load_data(train_data_path, norm=False)

    # MULTI LAYER PERCEPTRON
    layer_params = {"kernel_initializer":"glorot_uniform", "activation":"sigmoid"}
    compile_params = {"loss":"mean_squared_error", "optimizer":"rmsprop", "metrics":["accuracy"]}
    train_params = {"epochs":30, "batch_size":16, "verbose":0}

    mlp = Keras_Dense_MLP(neuron_layers=[len(train_data[0]), 1069, 199, 1], layer_params=layer_params, compile_params=compile_params, **train_params)

    def random_layer_gen(indim, outdim, hidden_ranges):
        fc = len(scaled_train_data[0])
        layers = [fc]
        layers.append(np.random.randint(fc*2,fc*15))
        add_alyer = np.random.choice([True,False])
        if(add_alyer):
            layers.append(np.random.randint(fc*2,fc*10))
        layers.append(1)
        return layers

    mlp_param_dist = {"epochs": [10*i for i in range(3,4)],
                      "batch_size": [np.power(2,i) for i in range(3,7)],
                      "neuron_layers": random_layer_gen,
                      "optimizer":["sgd","rmsprop"],
                      "loss":["mean_squared_error","mean_absolute_error"]}

    mlp_p_keys = ["epochs", "batch_size", "neuron_layers", "optimizer","loss"]
    mlp_p_keys_abrev = ["epochs", "b-size", "layers", "opt","loss"]

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
