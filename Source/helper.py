import numpy as np
#import json
import matplotlib.pyplot as plt
#import os
from sklearn.metrics import mean_absolute_error
#import time



def MAE(y_test, y_pred, Regressor = "Regressors"):
    """
    Compute and print the mean_absolute_error

    return: (MAE_acc, MAE_vox)

    """
    MAE_acc = mean_absolute_error(y_test.T[0].T, y_pred.T[0].T)
    MAE_vox = mean_absolute_error(y_test.T[1].T, y_pred.T[1].T)

    print(Regressor + " MAE_acc: " + str(MAE_acc))
    print(Regressor + " MAE_vox: " + str(MAE_vox))
    print(" ")

    return MAE_acc, MAE_vox



def plot(y_test, y_pred, subtitle="subtitle"):

    """
    plot the two predicted and groud truth loudness
    """

    t = np.arange(y_pred.shape[0])/10

    plt.figure()
    plt.suptitle(subtitle)

    plt.subplot(211)  #acc
    plt.title('Accompaniment Loudness compared to Mixture Loudness')
    plt.ylabel('short-term LUFS in dB')
    plt.xlabel('time in seconds')
    plt.plot(t, y_test.T[0], label="ground truth")
    plt.plot(t, y_pred.T[0], label="prediction")
    plt.legend(loc='lower center', ncol=2)


    plt.subplot(212)  #vox
    plt.title('Vocal Loudness compared to Mixture Loudness')
    plt.ylabel('short-term LUFS in dB')
    plt.xlabel('time in seconds')
    plt.plot(t, y_test.T[1], label="ground truth")
    plt.plot(t, y_pred.T[1], label="prediction")
    plt.legend(loc='lower center', ncol=2)

    plt.tight_layout(pad=1.0)

    plt.show()
