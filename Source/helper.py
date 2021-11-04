import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import random


def MAE(y_test, y_pred, Regressor = "unknown"):
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




def plot_histogram(y_test, y_pred, subtitle="subtitle", show_plot=False):

    """
    Plot the histogram of the error
    """

    abs_error = np.abs(y_test - y_pred)



    plt.figure()
    plt.suptitle(subtitle+"_error_histogram")

    ax = plt.subplot(211)  #acc

    n, bins, patches = ax.hist(abs_error.T[0], bins=100, density=1)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Accompaniment Loudness Absolute Error')


    ax = plt.subplot(212)  #vox


    n, bins, patches = ax.hist(abs_error.T[1], bins=100, density=1)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Vocal Loudness Absolute Error')



    plt.tight_layout(pad=1.0)


    plt.savefig("../Plots/New/" + subtitle + "_histogram" + '.png')

    if show_plot:
        plt.show()

    plt.close()






def plot(y_test, y_pred, subtitle="subtitle", show_plot=False, shuffle=False):

    """
    plot the two predicted and groud truth loudness
    """

    MAE_acc, MAE_vox = MAE(y_test, y_pred)
    ME_acc, ME_vox = ME(y_test, y_pred)

    if shuffle:
        stack = np.concatenate([y_test, y_pred], axis=1)
        np.random.shuffle(stack)
        y_test = stack.T[:2].T
        y_pred = stack.T[2:].T
        subtitle += "_shuffled"


    t = np.arange(y_pred.shape[0])/10

    plt.figure()
    plt.suptitle(subtitle)


    plt.subplot(211)  #acc
    plt.title('Accompaniment Loudness compared to Mixture Loudness' + "  MAE: " + str(MAE_acc))
    plt.ylabel('short-term LUFS in dB')
    plt.xlabel('time in seconds')
    plt.plot(t, y_test.T[0], label="ground truth")
    plt.plot(t, y_pred.T[0], label="prediction")
    plt.legend(loc='lower center', ncol=2)


    plt.subplot(212)  #vox
    plt.title('Vocal Loudness compared to Mixture Loudness' + " MAE: "+ str(MAE_vox))
    plt.ylabel('short-term LUFS in dB')
    plt.xlabel('time in seconds')
    plt.plot(t, y_test.T[1], label="ground truth")
    plt.plot(t, y_pred.T[1], label="prediction")
    plt.legend(loc='lower center', ncol=2)

    plt.tight_layout(pad=1.0)


    plt.savefig("../Plots/New/" + subtitle + '.png')

    if show_plot:
        plt.show()

    plt.close()




def plot_histogram_ground_truth(y):

    """
    Plot the histogram of the ground truth
    """

    fig, ax = plt.subplots()

    # the histogram of the data

    n, bins, patches = ax.hist(y.T[0], bins='auto', density=1)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Accompaniment Loudness')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


    fig, ax = plt.subplots()

    # the histogram of the data

    n, bins, patches = ax.hist(y.T[1], bins='auto', density=1)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Vocal Loudness')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()





def plot_histogram_error(error_matrix, show_plot=False, subtitle=""):

    """
    Plot the histogram of the error
    """

    plt.figure()
    plt.suptitle(subtitle+" Error Histogram (file level)")


    ax = plt.subplot(221)  #acc MAE

    n, bins, patches = ax.hist(error_matrix.T[0], bins=100)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Accompaniment Loudness Mean Absolute Error')


    ax = plt.subplot(222)  #vox MAE

    n, bins, patches = ax.hist(error_matrix.T[1], bins=100)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Vocal Loudness Mean Absolute Error')


    ax = plt.subplot(223)  #acc ME

    n, bins, patches = ax.hist(error_matrix.T[2], bins=100)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title('Accompaniment Loudness Maximum Error')


    ax = plt.subplot(224)  #vox ME

    n, bins, patches = ax.hist(error_matrix.T[3], bins=100)

    ax.set_xlabel('Loudness')
    ax.set_ylabel('Probability density')
    ax.set_title(' Vocal Loudness Maximum Error')

    plt.tight_layout(pad=1.0)



    plt.savefig("../Plots/Error_historgram/Total/File_Level/" + subtitle + "_histogram" + '.png')

    if show_plot:
        plt.show()

    plt.close()
