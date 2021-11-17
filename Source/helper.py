import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import random
from sklearn.preprocessing import StandardScaler

def preprocessing(X_train, y_train, X_test, y_test):

    y_train = np.where(y_train > 0, 0, y_train) #truncate everything above 0dB
    y_test = np.where(y_test > 0, 0, y_test)

    y_train = np.where(y_train < -15, -15, y_train) #truncate everything below -15dB
    y_test = np.where(y_test < -15, -15, y_test)

    #y_train = np.interp(y_train, (-15, 0), (0, 1))

    y_train = 10**(y_train/20) #convert dB to amplitude

    #y_train, X_train = uniform(y_train, X_train)

    #Normalization
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler



def uniform(y_train, X_train):
    """
    Because accompaniment relative loudness and vocal relative loudness is
    highly correlated, we only consider the distribution of accompaniment relative loudness

    force the subsampling close to a uniform distribution

    1. find the mean and var of the ground truth training data y_train
    2. identify the values of mean - var and mean + var (may have a scaling on var)
    3. sort the training data y_train
    4. randomly (uniform), random order index remove data points within the range of 2.
    5. randomly (normal distribution) remove data points from index.
    """
    train_stack = np.concatenate([y_train, X_train], axis=1)
    sorted_train_stack = train_stack[np.argsort(train_stack[:, 0])]

    #count, bins, ignored = plt.hist(sorted_train_stack.T[0], 1000, density=True)
    #plt.show()
    #plt.close()

    plot_histogram_ground_truth(y_train)


    mean = np.mean(y_train.T[1]) #acc
    var = np.var(y_train.T[1])
    size = y_train.T[1].shape[0]
    a = 1
    low_bound = mean - a * var
    high_bound = mean + a * var

    y_train_sorted = np.sort(y_train.T[1])

    low_bound_diff_array = np.abs(y_train_sorted - low_bound)
    low_bound_index = low_bound_diff_array.argmin()

    high_bound_diff_array = np.abs(y_train_sorted - high_bound)
    high_bound_index = high_bound_diff_array.argmin()

    mean_idx = (high_bound_index + low_bound_index)/2
    var_idx = (high_bound_index  - low_bound_index)/2

    print(low_bound)
    print(low_bound_index)

    print(high_bound)
    print(high_bound_index)


    idx2remove = np.random.normal(mean_idx, var_idx, size*2).astype(int) #1000 is how many data points to remove

    idx2remove = idx2remove[(idx2remove < size) & (idx2remove > 0)]
    #idx2remove = idx2remove[(idx2remove < (mean_idx+2*var_idx)) & (idx2remove > (mean_idx-2*var_idx))]


    sorted_train_stack_removed = np.delete(sorted_train_stack, idx2remove, 0)


    #count, bins, ignored = plt.hist(sorted_train_stack_removed.T[0], 1000, density=True)
    #plt.show()
    #plt.close()


    y_train = sorted_train_stack_removed.T[:2].T
    X_train = sorted_train_stack_removed.T[2:].T


    plot_histogram_ground_truth(y_train)

    print(y_train.shape)
    print(X_train.shape)

    return y_train, X_train





def MAE(y_test, y_pred, Regressor = "unknown"):
    """
    Compute and print the mean_absolute_error

    return: (MAE_acc, MAE_vox)

    """
    MAE_acc = mean_absolute_error(y_test.T[0].T, y_pred.T[0].T)
    MAE_vox = mean_absolute_error(y_test.T[1].T, y_pred.T[1].T)

    #print(Regressor + " MAE_acc: " + str(MAE_acc))
    #print(Regressor + " MAE_vox: " + str(MAE_vox))
    #print(" ")



    return MAE_acc, MAE_vox



def ME(y_test, y_pred, Regressor = "unknown"):
    """
    Compute and print the max_error

    return: (MAE_acc, MAE_vox)

    """
    ME_acc = max_error(y_test.T[0].T, y_pred.T[0].T)
    ME_vox = max_error(y_test.T[1].T, y_pred.T[1].T)

    #print(Regressor + " ME_acc: " + str(ME_acc))
    #print(Regressor + " ME_vox: " + str(ME_vox))
    #print(" ")

    return ME_acc, ME_vox




def plot_histogram(y_test, y_pred, subtitle="subtitle", show_plot=False):

    """
    Plot the histogram of the error
    """

    MAE_acc, MAE_vox = MAE(y_test, y_pred)
    ME_acc, ME_vox = ME(y_test, y_pred)

    abs_error = np.abs(y_test - y_pred)


    plt.figure()
    plt.suptitle(subtitle+"_error_histogram")

    ax_1 = plt.subplot(211)  #acc

    n, bins, patches = ax_1.hist(abs_error.T[0], bins=150, density=1)

    ax_1.set_xlim([0, 6])
    ax_1.set_ylim([0, 2])

    MAE_acc_str = "  Mean Absolute Error: " + str(MAE_acc)[:5] + "dB"
    ME_acc_str = "  Maximum Error: " + str(ME_acc)[:5] + "dB"
    ax_1.set_xlabel('Loudness in dB  \n ' + MAE_acc_str + "\n" + ME_acc_str)
    ax_1.set_ylabel('Probability density')

    ax_1.set_title('Histogram of Accompaniment Loudness Absolute Error')



    ax_2 = plt.subplot(212)  #vox

    n, bins, patches = ax_2.hist(abs_error.T[1], bins=150, density=1)

    ax_2.set_xlim([0, 6])
    ax_2.set_ylim([0, 2])

    MAE_vox_str = "  Mean Absolute Error: " + str(MAE_vox)[:5] + "dB"
    ME_vox_str = "  Maximum Error: " + str(ME_vox)[:5] + "dB"
    ax_2.set_xlabel('Loudness in dB \n ' + MAE_vox_str +  "\n" +   ME_vox_str)
    ax_2.set_ylabel('Probability density')
    ax_2.set_title('Histogram of Vocal Loudness Absolute Error')


    plt.tight_layout(pad=1.0)




    plt.savefig("../Plots/New/" + subtitle + "_histogram" + '.png')

    if show_plot:
        plt.show()

    plt.close()






def plot(y_test, y_pred, subtitle="subtitle", show_plot=False, shuffle=False):

    """
    plot the two predicted and ground truth loudness
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
    plt.title('Accompaniment Loudness compared to Mixture Loudness')
    plt.ylabel("short-term LUFS in dB")
    MAE_acc_str = "  Mean Absolute Error: " + str(MAE_acc)[:5] + "dB"
    ME_acc_str = "  Maximum Error: " + str(ME_acc)[:5] + "dB"
    plt.xlabel('time in seconds \n' + MAE_acc_str +  "\n" +   ME_acc_str)
    plt.plot(t, y_test.T[0], label="ground truth")
    plt.plot(t, y_pred.T[0], label="prediction")
    plt.legend(loc='lower center', ncol=2)


    plt.subplot(212)  #vox
    plt.title('Vocal Loudness compared to Mixture Loudness')
    plt.ylabel("short-term LUFS in dB")
    MAE_vox_str = "  Mean Absolute Error: " + str(MAE_vox)[:5] + "dB"
    ME_vox_str = "  Maximum Error: " + str(ME_vox)[:5] + "dB"
    plt.xlabel('time in seconds \n' + MAE_vox_str +  "\n" +   ME_vox_str)
    plt.plot(t, y_test.T[1], label="ground truth")
    plt.plot(t, y_pred.T[1], label="prediction")
    plt.legend(loc='lower center', ncol=2)

    plt.tight_layout(pad=1.0)


    plt.savefig("../Plots/New/" + subtitle + '.png')

    if show_plot:
        plt.show()

    plt.close()




def plot_histogram_ground_truth(y, title=""):

    """
    Plot the histogram of the ground truth
    """

    fig, ax = plt.subplots()

    # the histogram of the data

    n, bins, patches = ax.hist(y.T[0], bins=1000, density=1)

    ax.set_xlim([0, -20])


    ax.set_xlabel('Loudness in dB')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Accompaniment Loudness' + "_" + title)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    #plt.show()
    plt.savefig("../Plots/Ground_truth_historgram/" + title + "accompaniment" + '.png')
    plt.close()


    fig, ax = plt.subplots()

    # the histogram of the data

    n, bins, patches = ax.hist(y.T[1], bins=3000, density=1)

    ax.set_xlim([0, -20])

    ax.set_xlabel('Loudness in dB')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of Vocal Loudness'+ "_" + title)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    #plt.show()
    plt.savefig("../Plots/Ground_truth_historgram/" + title + "vocal" + '.png')

    plt.close()





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

    plt.tight_layout(pad=3.0)



    plt.savefig("../Plots/Error_historgram/Total/File_Level/" + subtitle + "_histogram" + '.png')

    if show_plot:
        plt.show()

    plt.close()
