import numpy as np
import json
import os
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import RegressorChain
import time
import helper
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


############################################################################

def data_creation(ground_truth_path, feature_path):
    """
    return:
        features: numpy matrix of NxF

        ground_truth_pair: numpy matrix of Nx2

        file_dict: dictionary of file names as keys and
            tuple of the index of the data points of the file as values
    """


    abs_ground_truth_path = os.path.abspath(ground_truth_path)

    file_dict = {}
    ground_truth_voxREL = np.array([])
    ground_truth_accREL = np.array([])


    abs_feature_path = os.path.abspath(feature_path)


    features = None
    unmatched = 0

    for filename in os.listdir(abs_ground_truth_path):

        if filename.endswith(".json"):

            #first_underscore = filename.find("_")
            #second_underscore = filename.find('_', first_underscore + 1)
            #third_underscore = filename.find('_', second_underscore + 1)

            #ground_truth_name = filename[:third_underscore] + "_features.json"

            ground_truth_name = filename[:-18] + "_features.json"

            f = open(abs_ground_truth_path+"/"+filename)
            ground_truth_dict = json.load(f)

            for key in list(ground_truth_dict.keys()):
                length = np.array(ground_truth_dict[key]).size

                if "_accREL_" in key:
                    current_ground_truth_accREL = np.array(ground_truth_dict[key])
                    ground_truth_accREL = np.concatenate([ground_truth_accREL, current_ground_truth_accREL], axis=None)

                if "_voxREL_" in key:
                    current_ground_truth_voxREL = np.array(ground_truth_dict[key])
                    ground_truth_voxREL = np.concatenate([ground_truth_voxREL, current_ground_truth_voxREL], axis=None)



            f_feature = open(abs_feature_path+"/"+ "mixture_" + ground_truth_name)
            feature_dict = json.load(f_feature)

            current_features = None
            for key in list(feature_dict.keys()):
                if current_features is None:
                    current_features = np.array(feature_dict[key])
                else:
                    current_features = np.concatenate([current_features, np.array(feature_dict[key])], axis=1)


            if features is None:
                features = current_features
            else:
                features = np.concatenate([features, current_features], axis=0)


            if length != current_features.shape[0]:
                print("   ")
                print(filename)
                print("feature length")
                print(current_features.shape[0])
                print("ground_truth_length")
                print(length)
                unmatched += 1
                continue

            file_dict[ground_truth_name] = (features.shape[0]-length, features.shape[0])


        ground_truth_pair = np.stack((ground_truth_accREL, ground_truth_voxREL), axis=-1)


    features = np.nan_to_num(features)
    ground_truth_pair = np.nan_to_num(ground_truth_pair)

    print("data created")

    return features, ground_truth_pair, file_dict


############################################################################

#ground_truth_path = "../Ground_truth/MIR-1K"
#feature_path = "../Features/MIR-1K"


ground_truth_path = "../Ground_truth/musdb18hq"
feature_path = "../Features/musdb18hq"

X, y, file_dict = data_creation(ground_truth_path, feature_path)


############################################################################

"""
Plot ground truth histogram
"""


helper.plot_histogram_ground_truth(y, "the complete dataset")


############################################################################


def Mean_learning(sub_X_train, sub_y_train, X_test, y_test, filename=""):

    """
    Mean value predictor

    Use the mean values of the training set groud truth as the low bound result
    """


    mean_values = np.mean(sub_y_train, axis=0)


    print("Mean value: " + str(mean_values))


    y_pred = np.zeros(y_test.shape)

    y_pred += mean_values


    MAE_acc, MAE_vox = helper.MAE(y_test, y_pred, filename+"_Mean_value")
    ME_acc, ME_vox = helper.ME(y_test, y_pred, filename+"_Mean_value")


    helper.plot(y_test, y_pred, filename+"_Mean_value")
    helper.plot_histogram(y_test, y_pred, filename+"_Mean_value")


    return (MAE_acc, MAE_vox, ME_acc, ME_vox), y_test, y_pred



############################################################################


"""
SVR
"""

from sklearn.svm import SVR

def SVR_learning(sub_X_train, sub_y_train, X_test, y_test, filename=""):
    """
    #SVR_chained_acc_first
    """

    regr = make_pipeline(SVR(C=1.0, epsilon=0.2))

    chain = RegressorChain(base_estimator=regr, order=[0, 1])

    chain.fit(sub_X_train, sub_y_train)

    y_pred = chain.predict(X_test)


    MAE_acc, MAE_vox = helper.MAE(y_test, y_pred, filename+"_SVR")
    ME_acc, ME_vox = helper.ME(y_test, y_pred, filename+"_SVR")

    helper.plot(y_test, y_pred, filename+"_SVR")

    helper.plot_histogram(y_test, y_pred, filename+"_SVR")


    return (MAE_acc, MAE_vox, ME_acc, ME_vox), y_test, y_pred


#SVR_learning(sub_X_train, sub_y_train, X_test, y_test)


def machine_learning(X, y, file_dict):
    """
    apply machine learning algorithms on each file
    """

    file_count = len(file_dict.keys())
    error_mean_matrix = np.zeros((file_count, 4))
    error_SVR_matrix = np.zeros((file_count, 4))
    idx = 0

    y_test_mean_total = None
    y_test_SVR_total = None
    y_pred_mean_total = None
    y_pred_SVR_total = None


    start = time.time()

    for filename in file_dict.keys():

        X_test = X[file_dict[filename][0] : file_dict[filename][1]]
        y_test = y[file_dict[filename][0] : file_dict[filename][1]]


        X_train = np.concatenate((X[: file_dict[filename][0]], X[file_dict[filename][1] :]), axis=0)
        y_train = np.concatenate((y[: file_dict[filename][0]], y[file_dict[filename][1] :]), axis=0)


        y_train, X_train = helper.uniform(y_train, X_train)

        print(y_train.size)

        sub_X_train = X_train[0:-1:60]
        sub_y_train = y_train[0:-1:60]

        #Normalization
        scaler = StandardScaler()
        scaler.fit(sub_X_train)

        sub_X_train = scaler.transform(sub_X_train)
        X_test = scaler.transform(X_test)

        error_mean, y_test_mean, y_pred_mean = Mean_learning(sub_X_train, sub_y_train, X_test, y_test, filename)
        error_SVR, y_test_SVR, y_pred_SVR = SVR_learning(sub_X_train, sub_y_train, X_test, y_test, filename)

        error_mean_matrix[idx] = error_mean
        error_SVR_matrix[idx] = error_SVR


        plot_error_histogram = True

        if plot_error_histogram:

            if y_pred_mean_total is None:
                y_test_mean_total = y_test_mean
                y_test_SVR_total = y_test_SVR
                y_pred_mean_total = y_pred_mean
                y_pred_SVR_total = y_pred_SVR
            else:
                y_test_mean_total = np.concatenate([y_test_mean_total, y_test_mean], axis=0)
                y_test_SVR_total = np.concatenate([y_test_SVR_total, y_test_SVR], axis=0)
                y_pred_mean_total = np.concatenate([y_pred_mean_total, y_pred_mean], axis=0)
                y_pred_SVR_total = np.concatenate([y_pred_SVR_total, y_pred_SVR], axis=0)

        #if idx >= 0: break
        idx += 1

    end = time.time()

    error_mean_matrix = error_mean_matrix[:idx]
    error_SVR_matrix = error_SVR_matrix[:idx]

    helper.plot_histogram(y_test_mean_total, y_pred_mean_total, "Total_Mean")
    helper.plot_histogram(y_test_SVR_total, y_pred_SVR_total, "Total_SVR")


    with open("../Results/file_list.json", 'w') as outfile:
        json.dump(list(file_dict.keys()), outfile)

    #json
    error_mean_list = error_mean_matrix.tolist()
    with open("../Results/error_mean.json", 'w') as outfile:
        json.dump(error_mean_list, outfile)
    #Binary data
    np.save('../Results/error_mean.npy', error_mean_matrix)
    #Human readable data
    np.savetxt('../Results/error_mean.txt', error_mean_matrix)

    #json
    error_SVR_list = error_SVR_matrix.tolist()
    with open("../Results/error_SVR.json", 'w') as outfile:
        json.dump(error_SVR_list, outfile)
    #Binary data
    np.save('../Results/error_SVR.npy', error_SVR_matrix)
    #Human readable data
    np.savetxt('../Results/error_SVR.txt', error_SVR_matrix)

    print("Total time:" + str(end - start) + "\n")

    return None


machine_learning(X, y, file_dict)


error_mean_matrix = np.load('../Results/error_mean.npy')
error_SVR_matrix = np.load('../Results/error_SVR.npy')


error_mean_value_average = np.mean(error_mean_matrix, axis=0)
error_SVR_average = np.mean(error_SVR_matrix, axis=0)

print("error_mean_value_average")
print(error_mean_value_average)
print("error_SVR_average")
print(error_SVR_average)

helper.plot_histogram_error(error_mean_matrix, subtitle="Mean Value")
helper.plot_histogram_error(error_SVR_matrix, subtitle="SVR")


quit()
