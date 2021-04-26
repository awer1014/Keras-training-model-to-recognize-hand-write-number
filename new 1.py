import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.datasets import mnist
from keras import regularizers
import keras.callbacks as CB
from keras.layers.normalization import BatchNormalization
import pandas as pd
import cv2          # 匯入 OpenCV 影像處理套件， 需要安裝
import glob         # 匯入內建的檔案與資料夾查詢套件
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape

def load_data():
    print ('Loading data...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(type(X_train))
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    X_test=[]
    y_test=[]
    for label in range(10):
        files = glob.glob( "/content/drive/MyDrive/handwrite/pic_2BPen/"+str(label)+"/*.png" )
        for file in files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #轉灰階
            img = cv2.bitwise_not(img)       # 反白：變成黑底白字
            img = cv2.resize(img, (28, 28))  # 重設大小為 28x28
            X_test.append(img)
            y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    X_train = np.expand_dims(X_train,axis=3)
    X_test = np.expand_dims(X_test,axis=3)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    print ('Data loaded.')
    return [X_train, X_test, y_train, y_test]

####################################################################
## You can setup different networks with different regularizers here
#####################################################################
def init_model(rms):
    start_time = time.time()
    print ('Compiling Model ... ')
    model = Sequential()
    model.add(Reshape((784,), input_shape=(None, 28, 28, 1)))
    model.add(Dense(500, kernel_regularizer=regularizers.l2(0.001))) #Use L2
    model.add(BatchNormalization())  #add a BN layer here
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))   #Use dropout
    #model.add(Dense(300, kernel_regularizer=regularizers.l1(0.001))) #Use L1
    model.add(BatchNormalization())  #add a BN layer here
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(BatchNormalization())  #add a BN layer here
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print ('Model compield in {0} seconds'.format(time.time() - start_time))
    return model

def run_network(rms, data=None, model=None, nb_epoch=20, batch_size=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model(rms)

        print ('Training model...')

        callbacks= [ CB.EarlyStopping(monitor='val_loss', patience=3, verbose=2) ] ## Use EarlyStop Here

        aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        horizontal_flip=True, fill_mode="nearest", validation_split=0.2)        

        history = model.fit_generator(aug.flow(X_train, y_train, batch_size=32),
              steps_per_epoch=len(X_train) // 32, epochs=2)

        print ("Training duration : {0}".format(time.time() - start_time))
        score = model.evaluate(X_test, y_test, batch_size=16)

        print ("Network's test score [loss, accuracy]: {0}".format(score))
        return model, history.history
    except KeyboardInterrupt:
        print (' KeyboardInterrupt')
        return model, history.history

def plot_losses(history_labels):
  for history, label in history_labels:
    plt.figure(1, figsize=(10,10)) 
    plt.subplot(211)   # 2 by 1 array of subplots, and draw the first one 
    plt.plot(history['loss'])  
    #plt.plot(history['val_loss'])  
    plt.title('model loss ' + label)  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper right')  
	
    plt.subplot(212)   # 2 by 1 array of subplots, and draw the second one 
    plt.plot(history['accuracy'])  
    #plt.plot(history['val_accuracy'])  
    plt.title('model accuracy ' + label)  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='lower right')  	
    plt.show() 
	
def plot_image(image):
    fig = plt.gcf() #design image size
    fig.set_size_inches(2,2) #design image size
    plt.imshow(image, cmap = 'binary') 
    plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig =plt.gcf()
    fig.set_size_inches(10,10)
    if num>25: num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1) #Generate 5*5 subgraph
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction)>0:
            title+=",predict=" + str(prediction[idx])
            ax.set_title(title, fontsize=10) #Setting subgraph title and size
            ax.set_xticks([]) #don't show the ticks
            ax.set_yticks([]) #don't show the ticks
            idx+=1
            plt.show()
	
#######################################################
## You can try different optimizers here by setting rms
########################################################

rms = Adam()
model3, hloss3 = run_network(rms)

labels = [ 'adam']

plot_losses(zip( [hloss3], labels) )	