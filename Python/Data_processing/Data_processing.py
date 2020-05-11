# Data processing for the data files created.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Definining paths:
username = 'hentze275'


# Importing the dataset
# dataset = pd.read_csv(".../telemetry/" + username + "/ks_vallelunga/tatuusfa1/132329983511571808.csv")
dataset = pd.read_csv("C:/UdpReceiver/telemetry/hentze275/ks_vallelunga/tatuusfa1/132329983511571808.csv")

# Checking dataset
rlap = dataset['rLap']
vCar = dataset['vCar_kph']

plt.plot(rlap,vCar)
plt.show()

plt.plot(dataset['sCarX'], dataset['sCarY'])
plt.show()

# Calculating sLap (Distance travelled)
for i in range(dataset.shape[0]):
    if i == 0:
        # if we want speed then delete np.power(x,2) and replace with x*x
        x_pos_delta = np.power(dataset['sCarX'][i-1] - dataset['sCarX'][i], 2)
        y_pos_delta = np.power(dataset['sCarY'][i-1] - dataset['sCarY'][i], 2)
        z_pos_delta = np.power(dataset['sCarZ'][i-1] - dataset['sCarZ'][i], 2)
        
        sLap[i]  = np.sqrt(x_pos_delta + y_pos_delta + z_pos_delta)
    else:
        sLap[i]  = 0
        

plt.plot(dataset['sCarX'],dataset['sCarY'])
plt.show()