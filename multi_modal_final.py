#!/usr/bin/env python
# coding: utf-8

# In[1]:

#import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import seaborn as sns
from tensorflow import keras


# In[2]:

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


# In[3]:


def cf(y_true, y_pred, fname, title):
    '''
    Create a confusion matrix plot save it into a file
    Inputs:
    y_true: actual labels
    y_pred: prediction labels
    fname: file name
    title: title of the plot
    Outputs:
    None
    '''
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    ax = sns.heatmap(cm, cmap= "Blues", annot=annot, fmt='')
    ax.set_title(title);
    ax.xaxis.set_ticklabels(['Normal','Attack'])
    ax.yaxis.set_ticklabels(['Normal','Attack'])
    plt.savefig(fname, dpi=300)
    plt.show()


#Load a single file as a numpy array
def load_file(filepath):
    data = np.load(filepath)
    return data



#Load network data
data1 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d1.npy")
data2 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d2.npy")
data3 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d28.npy")
data4 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d29.npy")
data5 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d30.npy")
data6 = load_file("/home/sepideh.bahadoripour/work/Data/Final_network/d31.npy")

#Delete unrelavant features like date, time, ...
data1= np.delete(data1,[0, 1, 2, 3],2 )
data2= np.delete(data2,[0, 1, 2, 3],2 )
data3= np.delete(data3,[0, 1, 2, 3],2 )
data4= np.delete(data4,[0, 1, 2, 3],2 )
data5= np.delete(data5,[0, 1, 2, 3, 20],2 )
data6= np.delete(data6,[0, 1, 2, 3, 20],2 )

#Concatenate data from different arrays into one array
data = np.concatenate((data1, data2, data3, data4, data5, data6), axis = 0)
#Reshape the data into 2D to pro-process it
data = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))

df = pd.DataFrame(data)
df.drop(13, axis = 1, inplace=True)

#Encode categorized features into integer values
labelencoder = LabelEncoder()
df[0]= labelencoder.fit_transform(df[0])
df[1]= labelencoder.fit_transform(df[1])
df[2]= labelencoder.fit_transform(df[2])
df[3]= labelencoder.fit_transform(df[3])
df[4]= labelencoder.fit_transform(df[4])
df[5]= labelencoder.fit_transform(df[5])
df[6]= labelencoder.fit_transform(df[6])
df[7]= labelencoder.fit_transform(df[7])
df[8]= labelencoder.fit_transform(df[8])
df[10]= labelencoder.fit_transform(df[10])
df[12]= labelencoder.fit_transform(df[12])

#Impute missing values using mean strategy
imputer = SimpleImputer(missing_values = np.NaN , strategy= 'mean') # Define the imputer with mean strategy
df = imputer.fit_transform(df) # Fit the imputer on the training data and transfer it

#Data normalization using MinMax method
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

#Reshape the data again to 3D for the LSTM model
data = np.reshape(df,(df.shape[0]//10,10,df.shape[1]))
data_net = np.asarray(data).astype('float32')


#Load sensor data
data_swat1 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/1.csv")
data_swat2 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/2.csv")
data_swat3 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/28.csv")
data_swat4 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/29.csv")
data_swat5 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/30.csv")
data_swat6 = pd.read_csv("/home/sepideh.bahadoripour/work/Data/Sensor/31.csv")

#Concatenate the data from different DataFrames into one
f = [data_swat1,data_swat2,data_swat3,data_swat4,data_swat5,data_swat6]
data_swat =pd.concat(f)
data_swat = data_swat.sample(data_net.shape[0])

#Extract the labels
y_swat = data_swat["Normal/Attack"]
data_swat = data_swat.drop(['Normal/Attack',' Timestamp'] , axis = 1)

#Encode the labels
y_swat[y_swat=='Normal']=0
y_swat[y_swat=='Attack']=1
y_swat[y_swat=='A ttack']=1

y_swat=np.array(y_swat)
y_swat=y_swat.astype("int64")

# data_swat = data_swat.sample(data_net.shape[0])
data_swat = np.array(data_swat)


# Make a unique dataset consists of both modalities to split
data = []
for i in range(data_net.shape[0]):
    data.append([data_net[i, :, :], data_swat[i, :]])

#Split the data into train, test, and validation parts
X_val, X_test, y_val, y_test = train_test_split(data, y_swat, stratify= y_swat, test_size= 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_val, y_val, stratify= y_val, test_size= 0.1)

#Extract network and sensor data to feed to the models
X_train_net = []
X_train_sen = []
for i in range (len(X_train)):
    X_train_net.append(X_train[i][0])
    X_train_sen.append(X_train[i][1])
    
X_test_net = []
X_test_sen = []
for i in range(len(X_test)):
    X_test_net.append(X_test[i][0])
    X_test_sen.append(X_test[i][1])

X_val_net = []
X_val_sen = []
for i in range(len(X_val)):
    X_val_net.append(X_val[i][0])
    X_val_sen.append(X_val[i][1])

#Convert final data to arrays
X_train_net = np.array(X_train_net)
X_test_net = np.array(X_test_net)
X_train_sen = np.array(X_train_sen)
X_test_sen = np.array(X_test_sen)
X_val_sen = np.array(X_val_sen)
X_val_net = np.array(X_val_net)


#Print the shape of data
print(X_train_net.shape)
print(X_test_net.shape)
print(X_val_net.shape)
print(X_train_sen.shape)
print(X_test_sen.shape)
print(X_val_sen.shape)

#One-hot vectoring the labels
y_train_oh = to_categorical(y_train, num_classes= 2)  
y_test_oh = to_categorical(y_test, num_classes= 2)
y_val_oh = to_categorical(y_val, num_classes= 2)

#Proposed deep multimodal model

#Sensor model
sens_inp = layers.Input(shape = (X_train_sen.shape[1]), name = "Sensor")
x = layers.Dense(512, activation = 'relu')(sens_inp)
x = layers.Dense(128, activation= 'relu')(x)
x = layers.Dense(256, activation = 'relu')(x)
sens_out = layers.Dense(32, activation = 'relu')(x)

#Network model
net_inp = layers.Input(shape = (X_train_net.shape[1], X_train_net.shape[2]), name = "Network")
x = layers.LSTM(1024, return_sequences= True)(net_inp)
x = layers.LSTM(256, return_sequences = True)(x)
# x = layers.Dense(512, activation = 'relu')(x)
# x = layers.Dense(64, activation = 'relu')(x)
net_out = layers.LSTM(128)(x)
# net_out = layers.Dense(32, activation= 'relu')(x)

#Fusion model
x = layers.concatenate([sens_out, net_out])
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dense(32, activation = 'relu')(x)

#Classification layer
out = layers.Dense(2, activation = 'softmax')(x)

#Final model
model = Model(inputs = [net_inp, sens_inp], outputs = out)

model.summary()

#Callbacks
model_name = "best_model.h5"
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 70)
monitor = keras.callbacks.ModelCheckpoint(model_name, monitor='val_accuracy', verbose=2 ,save_best_only=True,save_weights_only=True,mode='max')

#Compile and fit the model
model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit([X_train_net, X_train_sen], y_train_oh, epochs= 170, batch_size = 512, validation_data=([X_val_net, X_val_sen], y_val_oh), callbacks= [early_stop, monitor], verbose = 2)

#Get the prediction on test data
y_pred = np.argmax(model.predict([X_test_net, X_test_sen]), axis = 1)

#Create and save a confusion matrix
cf(y_test, y_pred, "mm_cf", "Multimodal Model")

#Print out the performance metrics
print(classification_report(y_test, y_pred))
