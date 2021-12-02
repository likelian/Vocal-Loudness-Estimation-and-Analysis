import numpy as np
#import json
import os
#from sklearn.pipeline import make_pipeline
#from sklearn.multioutput import RegressorChain
import time
import helper
import matplotlib.pyplot as plt
import pickle





def estimate(chain, X_test):

    scalerfile = '../Model/scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    X_test = scaler.transform(X_test)

    y_pred = chain.predict(X_test[1000:1200])

    #post-processing
    y_pred = np.where(y_pred <= 0.00001, 0.000001, y_pred)
    y_pred = np.where(y_pred >= 1, 1, y_pred)

    y_pred = 20 * np.log10(y_pred) #convert amplitude to dB
    y_pred = np.nan_to_num(y_pred, nan=-15, posinf=0, neginf=-15)


    return y_pred


#extract feature one by one?

# Load
X = np.load("../Data/"+"X.npy")
#y = np.load("../Data/"+"y.npy")
file_dict = np.load("../Data/"+"file_dict.npy",allow_pickle='TRUE').item()



# load
model_location = '../model/svr.pkl'
with open(model_location, 'rb') as f:
    chain = pickle.load(f)

estimate(chain, X)
