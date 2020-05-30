# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset = pd.read_csv('D:\GitHub\DataPostProcess-ML\Python\Models\SlipAngleModel\FirstGen\Data\LotusR12520LapsMugello.csv')

# Removing NaN
dataset = dataset.dropna()


# Creating the X variables
X = dataset.drop(columns= ["rTyreSlipFR",
                           "rTyreSlipFL",
                           "rTyreSlipRL",
                           "rTyreSlipRR",
                           "aTyreSlipFR",
                           "aTyreSlipFL",
                           "aTyreSlipRL",
                           "aTyreSlipRR",
                           "ndSlipFL",
                           "ndSlipFR",
                           "ndSlipRL",
                           "ndSlipRR",
                           "isInPit"])

# Creating the y variabel
y = dataset["rTyreSlipFR"].values
y = y.reshape(-1,1)

# Checking head
X_head = X.head()

# Encoding True/False into 0 and 1
arraynames = ['isAbsEnabled','isAbsInAction','isTcEnabled','isTcInAction','isEngineLimiterOn']
for name in arraynames:
    X[name] = X[name].astype(int)

# Checking head
X_head = X.head()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


# Testing with principal component analysis
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 19) # put None to check the variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
total = sum(explained_variance)



def build_regressor_constant():

    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout

    # Initialising the ANN
    regressor = Sequential()

    # Adding the input layer and the first hidden layer
    regressor.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu', 
                        input_dim = 19))
    regressor.add(Dropout(rate = 0.1))

    # Adding the second hidden layer
    regressor.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu'))
    regressor.add(Dropout(rate = 0.1))
    
    # Adding the second hidden layer
    regressor.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu'))
    regressor.add(Dropout(rate = 0.1))
    
    # Adding the second hidden layer
    regressor.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu'))
    regressor.add(Dropout(rate = 0.1))
    
    # Adding the second hidden layer
    regressor.add(Dense(units = 50, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu'))
    regressor.add(Dropout(rate = 0.1))
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the ANN
    regressor.compile(optimizer = 'adam', 
                      loss = 'mean_squared_error')
    
    # Fitting the ANN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    
    return regressor

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred)

plt.plot(y_test, color = 'red', label = 'test')
plt.plot(y_pred, color = 'blue', label = 'pred')
plt.xlabel('time')
plt.ylabel('slip_scaled')
plt.legend()
plt.show()

# Improving the ANN
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(nr_layers,neurons,optimizer,rate,epochs,batch_size):
    # Initialising the ANN
    regressor = Sequential()

    # Adding the input layer and the first hidden layer
    regressor.add(Dense(units = neurons, 
                        kernel_initializer = 'uniform', 
                        activation = 'relu', 
                        input_dim = 19))
    regressor.add(Dropout(rate = rate))
    
    for nr_layers
        # Adding the hidden layers as specified by nr_layers
        regressor.add(Dense(units = neurons, 
                            kernel_initializer = 'uniform', 
                            activation = 'relu'))
        regressor.add(Dropout(rate = rate))
    
    
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the ANN
    regressor.compile(optimizer = optimizer, 
                      loss = 'mean_squared_error')
    
    # Fitting the ANN to the Training set
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

    return regressor

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32,50],
              'nb_epoch' : [100,250,500],
              'optimizer' : ['adam', 'rmsprop'],
              'neurons' : [20,50,70],
              'rate' : [0.1,0.25,0.5],
              'nr_layers' : [5,10,12]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

