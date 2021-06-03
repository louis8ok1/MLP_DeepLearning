import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense, SpatialDropout2D, Dropout, Flatten
from  tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from  tensorflow.keras.optimizers import SGD,RMSprop
from  tensorflow.keras.callbacks import EarlyStopping
from  tensorflow.keras.losses import BinaryCrossentropy
from  tensorflow.keras.utils import to_categorical
from  tensorflow.keras.optimizers import Adam

def label_data():
    class_1 = [[0]]*5000
    class_2 = [[1]]*5000
    
    train_label = np.concatenate((class_1,class_2),axis=0)
    test_label = np.concatenate((class_1,class_2),axis=0)
    train_label = to_categorical(train_label,num_classes=2)
    test_label = to_categorical(test_label,num_classes=2)
    print('train_label')
    print(train_label)
    return train_label,test_label
def loss_pic(History):
    history = History
  
    history = History
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_data', 'val_data'], loc='upper left')
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig('./Model accuracy_{}.png'.format(timestr))
    plt.cla()
      
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_data', 'val_data'], loc='upper right')
    timestr = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig('./Model loss_{}.png'.format(timestr))
    plt.cla()
    #timestr = time.strftime("%Y%m%d_%H%M%S")
   
    
def Build_model(train_data, test_data, train_label, test_label):
    model=Sequential()
    #layer 1 
    model.add(Dense(256, input_dim=50, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    #layer 2 
    model.add(Dense(64, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    #layer 3 
    model.add(Dense(32, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
 
    model.add(Dense(2,activation='tanh'))
    
    
    model.compile(loss='BinaryCrossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    #24 85%
    #32 87%
    #48 86%
    history = model.fit(train_data, train_label, batch_size=32, epochs=500,
                        shuffle=True,
                        validation_split=0.2)
    History = history
    
    #model.evaluate(test_data,test_label, batch_size=24)
    #predict
    print('\nTesting------------')
    y_pred = model.predict(test_data)
    print(y_pred)
    for i in range(10000):
        if y_pred[i][0]>y_pred[i][1] :
            y_pred[i][0]=1
            y_pred[i][1]=0
        elif y_pred[i][0]<y_pred[i][1]:
            y_pred[i][0]=0
            y_pred[i][1]=1

    counter = 0
    for i in range(10000):
        
        if y_pred[i][0]!=test_label[i][0] or y_pred[i][1]!=test_label[i][1]:
            counter+=1
            

    print('Accuracy for test data:',(10000-counter)/10000)
    #draw loss picture
    loss_pic(History)
    #timestr = time.strftime("%Y%m%d_%H%M%S")
    #model.save('./model/model_{}.h5'.format(timestr)) 
        
if __name__ =="__main__":
    #load data
    train_data = np.load('./data/train_data.npy')
    test_data  = np.load('./data/test_data.npy')
    print(train_data[0].shape)
    #label
    train_label,test_label = label_data()
    #Build model
    
    Build_model(train_data ,test_data,train_label,test_label)