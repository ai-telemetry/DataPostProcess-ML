# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from pathlib import Path

# Importing the dataset #Run1LotusR125
Run1 = pd.read_csv("C:/Users/johan/Documents/GitHub/post_process_code/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run1LotusR125.csv")
Run2 = pd.read_csv("C:/Users/johan/Documents/GitHub/post_process_code/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run2LotusR125.csv")
Run3 = pd.read_csv("C:/Users/johan/Documents/GitHub/post_process_code/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run3LotusR125.csv")
Run4 = pd.read_csv("C:/Users/johan/Documents/GitHub/post_process_code/Python/Models/SlipAngleModel/LotusR125/Data/Barcelona/Run4LotusR125.csv")

# Attempt at using relative path
"""import csv
from pathlib import Path

base_path = Path('__file__').parent
file_path = (base_path / '../Data/Barcelona/Run1LotusR125.csv').resolve()

with open(file_path) as f:
    data_run1 = [line for line in csv.reader(f)]"""

# Concat the different test sets
dataset_up = pd.concat([Run1, Run2, Run3, Run4], axis=0)
dataset_heading = dataset_up.head()

# Pulling out relevan coloums for the slip angle
dataset = dataset_up[['vCar','gLat','gLong','aSteer','nWheelSpeedFR','dYDamperFR','aTyreCamberFR','hRideFR','aTyreSlipFR']]

# Removing sections where vCar is equal to zero
# Get names of indexes for which column vCar has value 0
indexNames = dataset[ dataset['vCar'] == 0 ].index
 
# Delete these row indexes from dataFrame
dataset.drop(indexNames , inplace=True)

# Dropping NA values from data
dataset = dataset.dropna(axis = 0)

# Creating sets for the analysis
X = dataset.iloc[:25000,:8].values
y = dataset.iloc[:25000, 8].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing Keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding first hidden layer
classifier.add(Dense(units=5,
                     activation='relu',
                     kernel_initializer='uniform',
                     input_dim=8))

# Adding second hidden layer
classifier.add(Dense(units=5,
                     activation='relu',
                     kernel_initializer='uniform'))

# Adding output layer
classifier.add(Dense(units=1,
                     activation='linear', # If more then 1 output submax
                     kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the ANN
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


