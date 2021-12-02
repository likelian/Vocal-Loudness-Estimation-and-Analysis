import numpy as np
import json
import os
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import RegressorChain
import time
import helper
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle

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
    abs_feature_path = os.path.abspath(feature_path)

    file_dict = {}
    ground_truth_voxREL = np.array([])
    ground_truth_accREL = np.array([])
    ground_truth_bandRMS = np.zeros((0,10))


    features = None
    unmatched = 0

    for filename in os.listdir(abs_ground_truth_path):

        if filename.endswith(".json"):

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


                if "vox_bandRMS" in key and "minus" not in key:
                    current_ground_truth_vox_abs_bandRMS = np.array(ground_truth_dict[key]).T

                if '_mix_bandRMS' in key:
                    current_ground_truth_mix_bandRMS = np.array(ground_truth_dict[key]).T

                if "acc_minus_vox_bandRMS" in key:
                    acc_minus_vox_bandRMS = np.array(ground_truth_dict[key]).T

            current_ground_truth_bandRMS = current_ground_truth_vox_abs_bandRMS - current_ground_truth_mix_bandRMS

            ground_truth_bandRMS = np.concatenate([ground_truth_bandRMS, current_ground_truth_bandRMS], axis=0)


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


            print(current_features.shape)
            print(current_ground_truth_accREL.shape)
            print(length)

            #print(ground_truth_pair.shape)


            if length/10 != current_features.shape[0]:
                print("   ")
                print(filename)
                print("feature length")
                print(current_features.shape[0])
                print("ground_truth_length")
                print(length)
                unmatched += 1
                #continue



            file_dict[ground_truth_name] = (features.shape[0]-int(length/10), features.shape[0])


        ground_truth_pair = np.concatenate((ground_truth_accREL[:, None], ground_truth_voxREL[:, None], ground_truth_bandRMS), axis=1)


    features = np.nan_to_num(features)
    ground_truth_pair = np.nan_to_num(ground_truth_pair)

    print("data created")

    return features, ground_truth_pair, file_dict


############################################################################


def Mean_training(sub_X_train, sub_y_train):

    """
    Mean value predictor
    Use the mean values of the training set groud truth as the low bound result
    """

    mean_values = np.mean(sub_y_train, axis=0)

    print("Mean value: " + str(mean_values))

    return mean_values



def Mean_fitting(mean_values, y_test):

    y_pred = np.zeros(y_test.shape)
    y_pred += mean_values

    y_pred = 20 * np.log10(y_pred) #convert amplitude to dB

    return y_pred



def eval(y_test, y_pred, filename="", model="_Mean_value"):
    """
    evaluate
    """

    MAE_acc, MAE_vox, MAE_bandRMS = helper.MAE(y_test, y_pred, filename+model)
    ME_acc, ME_vox, ME_bandRMS = helper.ME(y_test, y_pred, filename+model)

    test_nan_idx = np.argwhere(y_test.T[1]<=-14.999)
    y_test_nan = np.copy(y_test)
    y_test_nan[test_nan_idx] = np.nan
    y_test_mean = np.nanmean(y_test, axis=0)

    pred_nan_idx = np.argwhere(y_pred.T[1]<=-14.999)
    y_pred_nan = np.copy(y_pred)
    y_pred_nan[pred_nan_idx] = np.nan
    y_pred_mean = np.nanmean(y_pred, axis=0)


    error_mean = np.abs(y_test_mean - y_pred_mean)

    #y_test_no_nan = np.where()
    #y_test_mean = np.mean(y_test, axis=0)
    #y_pred_mean = np.mean(y_pred, axis=0)
    print("-------------")
    print(model)
    print("y_test_mean")
    print(y_test_mean)
    print("y_pred_mean")
    print(y_pred_mean)
    print("error_mean")
    print(error_mean)
    print("-------------")

    helper.plot(y_test, y_pred, filename+model)
    helper.plot_histogram(y_test, y_pred, filename+model)


    return (MAE_acc, MAE_vox, ME_acc, ME_vox, MAE_bandRMS, ME_bandRMS), y_test, y_pred



############################################################################


"""
SVR
"""

from sklearn.svm import SVR

def SVR_training(sub_X_train, sub_y_train):
    """
    #SVR_chained_acc_first
    """
    #{'C': 1, 'epsilon': 0.2, 'gamma': 0.001}  target acc
    regr = make_pipeline(SVR(C=1, epsilon=0.2, kernel='rbf'))

    chain = RegressorChain(base_estimator=regr, order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    chain.fit(sub_X_train, sub_y_train)

    return chain


def SVR_fitting(chain, X_test):

    y_pred = chain.predict(X_test)

    y_pred = np.where(y_pred <= 0.00001, 0.000001, y_pred)
    #y_test.T[:2] = np.where(y_test.T[:2] < -15, -15, y_test.T[:2])
    y_pred = np.where(y_pred >= 1, 1, y_pred)

    #y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1, neginf=0.01)
    y_pred = 20 * np.log10(y_pred) #convert amplitude to dB
    y_pred = np.nan_to_num(y_pred, nan=-15, posinf=0, neginf=-15)

    #y_pred = np.interp(y_pred, (0, 1), (-15, 0))

    return y_pred




def machine_learning_N_Fold(X, y, file_dict, extra=False, X_extra=None, y_extra=None):
    """
    apply machine learning algorithms on each file
    """

    file_count = len(file_dict.keys())
    error_mean_matrix = np.zeros((file_count, 24))
    error_SVR_matrix = np.zeros((file_count, 24))
    idx = 0

    y_test_mean_total = None
    y_test_SVR_total = None
    y_pred_mean_total = None
    y_pred_SVR_total = None


    start = time.time()

    if extra:

        X_extra, y_extra, o_o, o__o, scaler = helper.preprocessing(X_extra, y_extra, X_extra, y_extra)

        sub_X_train_extra = X_extra[::100]
        sub_y_train_extra = y_extra[::100]

        print("sub_X_train_extra")
        print(sub_X_train_extra.shape)

        start_MUSDB = time.time()
        mean_values = Mean_training(sub_X_train_extra, sub_y_train_extra)


        #chain = SVR_training(sub_X_train_extra, sub_y_train_extra)

        end_MUSDB = time.time()

        # save
        #with open('../model/svr.pkl','wb') as f:
        #    pickle.dump(chain,f)

        # load
        with open('../model/svr.pkl', 'rb') as f:
            chain = pickle.load(f)

    #quit()


    for filename in file_dict.keys():

        X_test = X[file_dict[filename][0] : file_dict[filename][1]]
        y_test = y[file_dict[filename][0] : file_dict[filename][1]]

        X_train = np.concatenate((X[: file_dict[filename][0]], X[file_dict[filename][1] :]), axis=0)
        y_train = np.concatenate((y[: file_dict[filename][0]], y[file_dict[filename][1] :]), axis=0)


        if not extra:
            X_train, y_train, X_test, y_test, scaler = helper.preprocessing(X_train, y_train, X_test, y_test)
            sub_X_train = X_train[::50]
            sub_y_train = y_train[::50]
        else:
            y_test = np.where(y_test > 0, 0, y_test)
            #y_test.T[:2] = np.where(y_test.T[:2] < -15, -15, y_test.T[:2])
            y_test = np.where(y_test < -15, -15, y_test)
            X_test = scaler.transform(X_test)


        if not extra:
            mean_values = Mean_training(sub_X_train, sub_y_train)
        y_pred = Mean_fitting(mean_values, y_test)



        error_mean, y_test_mean, y_pred_mean = eval(y_test, y_pred, filename, model="_Mean_value")
        error_mean_bandRMS = error_mean[-2:]
        error_mean = error_mean[:-2]

        #print(error_mean_bandRMS)

        if not extra:
            chain = SVR_training(sub_X_train, sub_y_train)

        y_pred = SVR_fitting(chain, X_test)



        error_SVR, y_test_SVR, y_pred_SVR = eval(y_test, y_pred, filename, model="_SVR")
        error_SVR_bandRMS = error_SVR[-2:]
        error_SVR = error_SVR[:-2]


        #print(error_SVR.shape)
        #print(error_SVR_bandRMS.shape)
        error_mean_all = np.append(np.asarray(error_mean, dtype=np.float32), error_mean_bandRMS)
        error_SVR_all = np.append(np.asarray(error_SVR, dtype=np.float32), error_SVR_bandRMS)

        error_mean_matrix[idx] = error_mean_all
        error_SVR_matrix[idx] = error_SVR_all



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

        if idx >= 30: break
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

    print("Total MUSDB time:" + str(end_MUSDB - start_MUSDB) + "\n")


    return None




############################################################################

ground_truth_path = "../Ground_truth/MIR-1K"
feature_path = "../Features/MIR-1K"


#X, y, file_dict = data_creation(ground_truth_path, feature_path)

#np.save("../Data/"+"X.npy", X)
#np.save("../Data/"+"y.npy", y)
#np.save("../Data/"+"file_dict.npy", file_dict)

# Load
X = np.load("../Data/"+"X.npy")
y = np.load("../Data/"+"y.npy")
file_dict = np.load("../Data/"+"file_dict.npy",allow_pickle='TRUE').item()


############################################################################

"""
Plot ground truth histogram
"""


helper.plot_histogram_ground_truth(y, "the complete dataset")

############################################################################


ground_truth_path = "../Ground_truth/musdb18hq"
feature_path = "../Features/musdb18hq"

#X_extra, y_extra, file_dict_extra = data_creation(ground_truth_path, feature_path)

#np.save("../Data/"+"X_extra.npy", X_extra)
#np.save("../Data/"+"y_extra.npy", y_extra)
#np.save("../Data/"+"file_dict_extra.npy", file_dict_extra)

# Load
X_extra = np.load("../Data/"+"X_extra.npy")
y_extra = np.load("../Data/"+"y_extra.npy")
#file_dict_extra = np.load("../Results/"+"file_dict_extra.npy",allow_pickle='TRUE').item()


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    #epsilons = [0.1,0.2,0.3,0.5]
    epsilons = [0.1]
    param_grid = {'C': Cs, 'gamma' : gammas, 'epsilon' : epsilons}
    y = y.T[1].T

    X = X[::1000]
    y = y[::1000]
    print(X.shape)
    print(y.shape)
    #quit()
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_params_
#svc_param_selection(X_extra, y_extra, 10)



with open('../model/svr.pkl', 'rb') as f:
    chain = pickle.load(f)

#X_train, y_train, X_test, y_test, scaler = helper.preprocessing(X, y, X, y)

machine_learning_N_Fold(X, y, file_dict, True, X_extra, y_extra)

#machine_learning_N_Fold(X, y, file_dict)



############################################################################


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
