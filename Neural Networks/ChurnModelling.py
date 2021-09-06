# We will try to model the churn of customers and whether the given customer will churn or not
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")

#data preprocessing

x=data.iloc[:,3:13].values
y=data.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labelencoder_X_1 = LabelEncoder()
x[:,1]=labelencoder_X_1.fit_transform(x[:,1])
labelencoder_X_2= LabelEncoder()
x[:,2]=labelencoder_X_2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x= onehotencoder.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)



#defining a callback
class mycallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.865):
            self.model.stop_training= True
            
callbackser= mycallback()

import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(output_dim= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim= 11 ))
model.add(Dense(output_dim= 6, kernel_initializer = 'uniform', activation = 'relu' ))
model.add(Dense(output_dim= 1, kernel_initializer = 'uniform', activation = 'sigmoid' ))
model.compile(optimizer= 'sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=10, epochs=100,callbacks=[callbackser])

# taking 0.5 as cutoff value for true/false classification
y_pred= model.predict(x_test)
y_pred= (y_pred>0.5)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plotting confusion matrix

import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

