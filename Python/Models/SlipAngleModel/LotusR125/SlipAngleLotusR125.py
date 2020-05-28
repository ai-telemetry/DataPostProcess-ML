# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from pathlib import Path

# Importing the dataset #Run1LotusR125
Run1 = pd.read_csv("D:/GitHub/DataPostProcess-ML/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run1LotusR125.csv")
Run2 = pd.read_csv("D:/GitHub/DataPostProcess-ML/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run2LotusR125.csv")
Run3 = pd.read_csv("D:/GitHub/DataPostProcess-ML/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run3LotusR125.csv")
Run4 = pd.read_csv("D:/GitHub/DataPostProcess-ML/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run4LotusR125.csv")
#Run4 = pd.read_csv("D:/GitHub/DataPostProcess-ML/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/f4.csv")


# Attempt at using relative path
"""import csv
from pathlib import Path

base_path = Path('__file__').parent
file_path = (base_path / '../Data/Barcelona/Run1LotusR125.csv').resolve()

with open(file_path) as f:
    data_run1 = [line for line in csv.reader(f)]"""

# Concat the different test sets - Run1, Run2,
dataset_up = pd.concat([Run1, Run2, Run3, Run4], axis=0)

# Pulling out relevan coloums for the slip angle
dataset = dataset_up[['vCar','gLat','gLong','gVert','aSteer','hCoG','nWheelSpeedFR','dYDamperFR','fZTyreFR','nTyreDirtLevelFR','aTyreCamberFR','tyreLoadedRadiusFR','hRideRL','hRideRR','hRideFL','hRideFR','sCarX','sCarY','sCarZ','aTyreSlipFR']]

# Removing sections where vCar is equal to zero
# Get names of indexes for which column vCar has value 0
indexNames = dataset[ dataset['vCar'] > 0.5 ].index
dataset.drop(indexNames , inplace=True)
#indexNames2 = dataset[ np.abs( dataset['aTyreSlipFR'] ) > 10 ].index
#dataset.drop(indexNames2 , inplace=True)
indexNames3 = dataset[ np.abs( dataset['nTyreDirtLevelFR'] ) > 0 ].index
dataset.drop(indexNames3 , inplace=True)

# Dropping NA values from data
dataset = dataset.dropna(axis = 0)

dataset_heading = dataset.head()

n = dataset.shape[1]-1
# Creating sets for the analysis
X = dataset.iloc[:,:n].values 
y = dataset.iloc[:, n].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#y = StandardScaler()
#y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,)
"""# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # put None to check the variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
                                              random_state = 0)
"""

# Importing Keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding first hidden layer
classifier.add(Dense(units=32,
                     activation='relu',
                     kernel_initializer='uniform',
                     input_dim= n))

# Adding second hidden layer
classifier.add(Dense(units = 32,
                     activation='relu',
                     kernel_initializer='uniform'))

# Adding second hidden layer
classifier.add(Dense(units = 32,
                     activation='relu',
                     kernel_initializer='uniform'))

# Adding second hidden layer
classifier.add(Dense(units = 32,
                     activation='relu',
                     kernel_initializer='uniform'))



# Adding output layer
classifier.add(Dense(units=1,
                     activation='sigmoid', # If more then 1 output submax
                     kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'mean_squared_error',
                   metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
classifier.summary()

# Predicting the Test set results
y_pred = classifier.predict(X_test)



