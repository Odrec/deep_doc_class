import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import csv
#import filemvr

class ConvNet:

    def __init__(self):

        # path to the model weights files.
        weights_path = 'vgg16_weights.h5'
        #top_model_weights_path = 'first_try.h5'
        # dimensions of our images.
        img_width, img_height = 150, 150

        train_data_dir = 'data/train'
        validation_data_dir = 'data/validation'
        nb_train_samples = 2000
        nb_validation_samples = 800
        nb_epoch = 50

        # build the VGG16 network
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
#        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
      #  top_model.load_weights("C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\first_try.h5")

        # add the model on top of the convolutional base
        model.add(top_model)

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        #for layer in model.layers[:25]:
        for layer in model.layers[:]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        self.model = model
        self.model.load_weights('first_try.h5')

    def get_function(self,filepointer,metapointer=None):
        pass

    def train(self,data,labels):
        nb_epoch = 50
        nb_validation_sambles = 400
        nb_train_samples = 4000
        train_data_dir = r"Data\Train"
        train_validation_data_dir = r"Data\Validation"

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=32,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            train_validation_data_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

        self.model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_sambles)
        self.model.save_weights('first_try.h5')
        pass

    def get_confusion_matrix(self):
        nb_epoch = 50
        nb_validation_sambles = 400
        nb_train_samples = 4000
        train_data_dir = r"Data\Train"
        train_validation_data_dir = r"Data\Validation"
        train_datagen = ImageDataGenerator(rescale=1./255)
        fp = open('classification.csv','w+')
        wr = csv.writer(fp,delimiter=';')
        wr.writerow(['filename','class','prediction','proba'])

        #train_datagen = ImageDataGenerator(
        #    rescale=1./255,
        #    shear_range=0.2,
        #    zoom_range=0.2,
        #    horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=1,
            class_mode='binary',
            shuffle=False)
        tru_pos = 0
        fal_pos = 0
        tru_neg = 0
        fal_neg = 0
        #for i in range(train_datagen.N):
        N = 7854
        c = 0
        names = None
        for i in os.walk(r'.\Data\Train'):
            c += 1
            if c == 1:
                continue
            if names is None:
                names = i[2]
            else:
                names += i[2]
        print(names)
        for i in range(N):
            print(str(i)+'/'+str(N))
            file = train_generator.next()
            y = self.model.predict_classes(file[0],batch_size=1,verbose=0)
            proba = self.model.predict_proba(file[0],batch_size=1,verbose=0)
            #print(file[1])
            wr.writerow([names[i],int(file[1]),int(y),proba])
            if file[1] == 1.0:
                if y == 1.0:
                    tru_pos += 1
                else:
                    fal_neg += 1

            else:
                if y == 1.0:
                    fal_neg += 1
                else:
                    tru_neg += 1
        print('\t\tPos\tNeg')
        print('True\t'+str(tru_pos)+'\t'+str(tru_neg))
        print('False\t'+str(fal_pos)+'\t'+str(fal_neg))
        print()
        print('\t\tPos\tNeg')
        print('True\t'+str(tru_pos/N)+'\t'+str(tru_neg/N))
        print('False\t'+str(fal_pos/N)+'\t'+str(fal_neg/N))






        #predictions = self.model.evaluate_generator(train_generator,8)
        #print(self.model.metrics_names)
        #print(predictions)

def false_positive(y_true, y_pred):
    pass

import os
import random

def prepare(limit=200):
    limit = limit - 1
    t_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train"
    v_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation"
    pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 199
    while(True):
        if n > limit and p > limit:
            break
        if n <= limit:
            while True:
                f = random.choice(neg)
                if f in v_neg:
                    continue
                v_neg.append(f)
                break
            n += 1
        if p <=limit:
            while True:
                f = random.choice(pos)
                if f in v_pos:
                    continue
                v_pos.append(f)
                break
            p += 1
    for pos in v_pos:
        os.rename(t_path+'\\pos\\'+pos,v_path+'\\pos\\'+pos)
    for neg in v_neg:
        os.rename(t_path+'\\neg\\'+neg,v_path+'\\neg\\'+neg)
    return

def undo():
    v_pos,v_neg = get_val_files()

    t_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train"
    v_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation"

    for pos in v_pos:
        os.rename(v_path+'\\pos\\'+pos,t_path+'\\pos\\'+pos)
    for neg in v_neg:
        os.rename(v_path+'\\neg\\'+neg,t_path+'\\neg\\'+neg)
    pass

def get_files():
    neg = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train\neg")
    pos = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train\pos")
    return pos, neg

def get_val_files():
    neg = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation\neg")
    pos = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation\pos")
    return pos, neg

c = ConvNet()
#prepare(200)
#c.train(list(),list())
c.get_confusion_matrix()
#undo()

