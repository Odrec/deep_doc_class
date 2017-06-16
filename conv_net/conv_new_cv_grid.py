import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy.misc import imresize
import os
import shutil
import random
import csv
from fnmatch import fnmatch

grid_path = '/net/store/ni/projects/DeepDocClass/convnet/'

def setup(csv_file,v_path = grid_path+"../Data/Validation"):
    t_path = grid_path+"../Data/Train"
    #pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 1
    with open(csv_file,'r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        for row in rdr:
            if row[1] == '1':
                v_pos.append(row[0])
            elif row[1] == '0':
                v_neg.append(row[0])
    for pos in v_pos:
        for file in os.listdir(t_path+'/pos/'):
            if fnmatch.fnmatch(file,pos+'*'):
                type(file)
                shutil.move(t_path+'/pos/'+file,v_path+'/pos/'+file)
    for neg in v_neg:
        for file in os.listdir(t_path+'/neg/'):
            if fnmatch.fnmatch(file,neg+'*'):
                shutil.move(t_path+'/neg/'+file,v_path+'/neg/'+file)
    return

def undo_setup(csv_file,v_path = "./Data/Validation"):
    t_path = "./Data/Train"
    #pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 1
    with open(csv_file,'r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        for row in rdr:
            if row[1] == '1':
                v_pos.append(row[0])
            elif row[1] == '0':
                v_neg.append(row[0])
    for pos in v_pos:
        for file in os.listdir(v_path+'\\pos\\'):
            if fnmatch.fnmatch(file,pos+'*'):
                type(file)
                shutil.move(v_path+'\\pos\\'+file,t_path+'\\pos\\'+file)
    for neg in v_neg:
        for file in os.listdir(v_path+'\\neg\\'):
            if fnmatch.fnmatch(file,neg+'*'):
                shutil.move(v_path+'\\neg\\'+file,t_path+'\\neg\\'+file)
    return

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
        model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(2, 2)))
        """

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
        """

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
       # top_model.load_weights("C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\third_try.h5")

        # add the model on top of the convolutional base
        model.add(top_model)

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        #for layer in model.layers[:25]:
        for layer in model.layers[:]:
            layer.trainable = True

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        self.model = model
        #for simplicity reasons later on
        self.model.save(grid_path+'second_try.h5')
        self.model.load_weights(grid_path+'second_try.h5')

    def reload_weights(self,name):
        self.model.load_weights(name)

    def eval(self,x,y,batch_size=32,epoch=50,save_path=grid_path+'weights.h5'):
        try:
            self.model.load_weights(save_path)
        except:
            pass
        data = list()
        for i,pic in enumerate(x):
            img = load_img(pic)
            img = imresize(img,size=(150,150))
            xi = img_to_array(img).reshape(3,150,150)
            data.append(xi)
        l_d = len(data)
        data = np.array(data)
        #data = xi.reshape((l_d,)+data.shape)
        res = self.model.evaluate(data,y,batch_size=batch_size)
        self.model.save(save_path)
        return res

    def eval_generator(self,path,batch_size=32,epoch=50,save_path=grid_path+'weights.h5'):
        eval_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        eval_gen = eval_datagen.flow_from_directory(
            path,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=64,
            shuffle=True,
            class_mode='binary')
        self.model.evaluate_generator(eval_gen)
        self.model.save(save_path)

    def fit(self,x,y,batch_size=32,epoch=50,save_path=grid_path+'weights.h5'):

        try:
            self.model.load_weights(save_path)
        except:
            pass

        data = list()
        for i,pic in enumerate(x):
            img = load_img(pic)
            #img.show()
            img = imresize(img,size=(150,150))
            img = img.transpose()
            xi = img_to_array(img)#.reshape(3,150,150)

            data.append(xi)
        l_d = len(data)
        data = np.array(data)
        y = np.array(y)
        #data = xi.reshape((l_d,)+data.shape)
        self.model.fit(data,y,batch_size=batch_size,nb_epoch=epoch,shuffle=True)
        self.model.save(save_path)


    def fit_generator(self,path,batch_size=64,epoch=50,save_path=grid_path+'weights.h5'):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            path,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=64,
            shuffle=True,
            class_mode='binary')
        self.model.fit_generator(train_generator,nb_epoch=epoch,batch_size=batch_size)
        self.model.save(save_path)

    def fit_and_eval_with_generator(self,path,eval_path,batch_size=64,epoch=50,save_path='weights.h5'):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            path,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=64,
            shuffle=True,
            class_mode='binary')

        test_generator = train_datagen.flow_from_directory(
            eval_path,  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=64,
            shuffle=True,
            class_mode='binary')


        self.model.fit_generator(train_generator,nb_epoch=epoch,validation_data=test_generator,samples_per_epoch=2000,nb_val_samples=2000)
        self.model.save(save_path)


    def find_files(self,name,path,label,num_pages=5):
        """
        finds all pictures in the directory with a specific file ID
        :param name:
        :param path:
        :param label:
        :param num_pages:
        :return:
        """
        if label == '1':
            dir = '/pos/'
        else:
            dir = '/neg/'
        i = 0
        files = list()
        for file in os.listdir(path+dir):
            if fnmatch(file, name+'*'):
                files.append(path+dir+file)
                i += 1
        if i <= 5:
            return files
        else:
            result = list()
            result.append(files[0])
            result.append(files[-1])
            files.remove(files[0])
            files.remove(files[-1])
            for i in range(num_pages-2):
                f = random.choice(files)
                result.append(f)
                files.remove(f)
            return result

    def csv_to_filenames(self,path,file,num_pages=5):
        """
        creates data and labels from a single bin
        :param path:
        :param file:
        :param num_pages:
        :return:
        """
        data = list()
        labels = list()
        with open(file,'r') as fp:
            rdr = csv.reader(fp,delimiter=',')
            for line in rdr:
                files = self.find_files(line[0],path,line[1],num_pages=num_pages)
                n_pages = len(files)
                for i in range(n_pages):
                    labels.append(int(line[1]))
                data += files
        return data,labels

    def fit_on_bins(self,trainbins,valbin,num_pages=5,batch=32,epoch=50):
        """

        :param trainbins:
        :param valbin:
        :param num_pages:
        :return:
        """
        bin_name = "cvset-"
        end = '.csv'
        #train on 9 bins
        X = list()
        y = list()
        for i in range(10):
            if i in trainbins:
                data,labels = self.csv_to_filenames(grid_path+"../Data/"+"cvset"+str(i),grid_path+bin_name+str(i)+end,num_pages=num_pages)
                X += data
                y += labels
            print("Extracting bin: "+str(i))
        self.fit(X,y,batch,epoch,grid_path+"eval-"+str(valbin)+'.h5')
                #self.fit_and_eval_with_generator(".\\Data\\"+"cvset"+str(i),".\\Data\\"+"cvset"+str(valbin),64,1,"eval-"+str(valbin)+'.h5')
        data,labels = self.csv_to_filenames(grid_path+"../Data/"+"cvset"+str(valbin),bin_name+str(valbin)+end,num_pages=num_pages)
        result = self.eval(data,labels,batch,epoch,grid_path+"eval-"+str(valbin)+'.h5')
        print(result)
        self.reload_weights(grid_path+"second_try.h5")
        return result

    def cross_val(self):
        print("Starting setup...")
        for i in range(10):
            name = 'cvset'+str(i)
            #os.mkdir('.\\Data\\'+name)
            #os.mkdir('.\\Data\\'+name+'\\pos')
            #os.mkdir('.\\Data\\'+name+'\\neg')
            setup(grid_path+'cvset-'+str(i)+'.csv',grid_path+'../Data/'+name)
            print("Bin",i,"done")
        bins = [0,1,2,3,4,5,6,7,8,9]
        results = list()
        for i in range(10):
            valbin = i
            train = bins[:]
            train.remove(i)
            results.append(self.fit_on_bins(trainbins=train,valbin=valbin,num_pages=5,batch=32,epoch=10))
        with open(grid_path+'results.txt','w+') as f:
            wrtr = csv.writer(f,delimiter=',', lineterminator='\n')
            wrtr.writerow(self.model.metrics_names)
            max = 0.0
            max_i = 0
            for i in range(10):
                try:
                    if results[i][1] > max:
                        max = results[i][1]
                        max_i = i
                except:
                    pass
                wrtr.writerow(results[i])
            print(max,max_i)

    def create_k_models_with_validation_split(self,num_models=1,prefix='model',epochs=50,batch_size=64):
        for i in range(10):
            name = 'cvset'+str(i)
            #os.mkdir('.\\Data\\'+name)
            #os.mkdir('.\\Data\\'+name+'\\pos')
            #os.mkdir('.\\Data\\'+name+'\\neg')
            undo_setup(grid_path+'cvset-'+str(i)+'.csv',grid_path+'../Data/'+name)



c = ConvNet()
c.cross_val()
