import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import random



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



def ME(y_test, y_pred, Regressor = "Regressors"):
    """
    Compute and print the max_error

    return: (MAE_acc, MAE_vox)

    """
    ME_acc = max_error(y_test.T[0].T, y_pred.T[0].T)
    ME_vox = max_error(y_test.T[1].T, y_pred.T[1].T)

    print(Regressor + " ME_acc: " + str(ME_acc))
    print(Regressor + " ME_vox: " + str(ME_vox))
    print(" ")

    return ME_acc, ME_vox





def plot(y_test, y_pred, subtitle="subtitle", show_plot=False, shuffle=False):

    """
    plot the two predicted and groud truth loudness
    """


    if shuffle:
        stack = np.concatenate([y_test, y_pred], axis=1)
        np.random.shuffle(stack)
        y_test = stack.T[:2].T
        y_pred = stack.T[2:].T


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


    plt.savefig("../Plots/" + subtitle + '.png')

    if show_plot:
        plt.show()
