# Import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.ticker import PercentFormatter



def plot_histogram_file(ave_error_SVR_matrix, subtitle="subtitle", show_plot=False):

    """
    Plot the histogram of the error file level
    """

    MAE_acc = np.mean(ave_error_SVR_matrix.T[0])
    ME_acc  = np.max(ave_error_SVR_matrix.T[0])

    MAE_vox = np.mean(ave_error_SVR_matrix.T[1])
    ME_vox  = np.max(ave_error_SVR_matrix.T[1])

    print("MAE_acc, ME_acc, MAE_vox, ME_vox")
    print(MAE_acc, ME_acc, MAE_vox, ME_vox)

    MAE_acc_str = "  Mean Absolute Error: " + str(MAE_acc)[:5] + "dB"
    ME_acc_str = "  Maximum Error: " + str(ME_acc)[:5] + "dB"

    MAE_vox_str = "  Mean Absolute Error: " + str(MAE_vox)[:5] + "dB"
    ME_vox_str = "  Maximum Error: " + str(ME_vox)[:5] + "dB"


    df_acc = pd.DataFrame(ave_error_SVR_matrix.T[0], columns = ["a"])
    df_vox = pd.DataFrame(ave_error_SVR_matrix.T[1], columns = ["a"])


    f, axes = plt.subplots(2, 1,  constrained_layout = True)

    plt.suptitle("Average Loudness Estimation Error Histogram (file Level)")

    sns.set(font_scale=1.2)

    sns.histplot(x= "a", data=df, stat="probability", ax=axes[0], bins=30, kde=True)
    sns.histplot(x= "a", data=df, stat="probability", ax=axes[1], bins=30, kde=True)


    axes[0].set_title('Accompaniment Loudness Estimation Error', y=1.08)
    axes[1].set_title('Vocal Loudness Estimation Error', y=1.08)

    axes[0].set_xlim(reversed(axes[0].set_xlim()))
    axes[1].set_xlim(reversed(axes[1].set_xlim()))

    axes[0].set_xlim([0, 6])
    axes[0].set_ylim([0, 0.1])

    axes[1].set_xlim([0, 6])
    axes[1].set_ylim([0, 0.1])


    axes[0].set_xlabel("Short-term LUFS Error in dB \n " +
                        "\n" +
                        MAE_acc_str +
                        ME_acc_str)
    axes[0].set_ylabel('Percentage')

    axes[1].set_xlabel("Short-term LUFS Error in dB \n " +
                        "\n" +
                        MAE_vox_str +
                        ME_vox_str)
    axes[1].set_ylabel('Percentage')


    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))


    f.set_size_inches(8, 6)


    plt.savefig("../Plots/New/" + subtitle + "_histogram" + '.png')

    if show_plot:
        plt.show()

    plt.close()
