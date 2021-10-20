import numpy as np
import json
import os
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import RegressorChain
import time
import helper




############################################################################

def data_creation():
    """
    return:
        features, numpy matrix of NxF
        ground_truth_pair:  numpy matrix of Nx2
    """

    ground_truth_path = "../Ground_truth/"
    abs_ground_truth_path = os.path.abspath(ground_truth_path)


    ground_truth_voxREL = np.array([])
    ground_truth_accREL = np.array([])

    feature_path = "../Features/"
    abs_feature_path = os.path.abspath(feature_path)


    features = None
    unmatched = 0

    for filename in os.listdir(abs_ground_truth_path):

        if filename.endswith(".json"):

            first_underscore = filename.find("_")
            second_underscore = filename.find('_', first_underscore + 1)

            ground_truth_name = filename[:second_underscore] + "_features.json"


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


            f_feature = open(abs_feature_path+"/"+ground_truth_name)
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

        ground_truth_pair = np.stack((ground_truth_accREL, ground_truth_voxREL), axis=-1)


    features = np.nan_to_num(features)
    ground_truth_pair = np.nan_to_num(ground_truth_pair)

    print("data created")

    return features, ground_truth_pair


############################################################################

X, y = data_creation()

############################################################################


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_size = X_train.shape[0]
test_size = X_test.shape[0]

print("train_size")
print(train_size)

print("test_size")
print(test_size)

############################################################################


#subsample

"""
The above data split are ignored
"""


sub_X_train = X_train[0:train_size:5]
sub_y_train = y_train[0:train_size:5]

print("sub_train_size")
print(sub_X_train.shape)


############################################################################



print("split before 1000 and after 1000")
sub_X_train = X[1000:][0:-1:10]
sub_y_train = y[1000:][0:-1:10]
X_test = X[:1000]
y_test = y[:1000]
print(sub_X_train.shape)



############################################################################

"""
Normalization
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(sub_X_train)

sub_X_train = scaler.transform(sub_X_train)
X_test = scaler.transform(X_test)


############################################################################

"""

Use the mean values of the training set groud truth as the low bound result

"""

print(sub_y_train.shape)



mean_values = np.mean(sub_y_train, axis=0)


print("Mean value: " + str(mean_values))


y_pred = np.zeros(y_test.shape)

y_pred += mean_values


helper.MAE(y_test, y_pred, "Mean Value")

helper.plot(y_test, y_pred, "Mean Value")



############################################################################


"""
SVR
"""

from sklearn.svm import SVR


regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[0, 1])

start = time.time()

chain.fit(sub_X_train, sub_y_train)

end = time.time()

print("SVR training time: " + str(end - start) + "\n")


y_pred = chain.predict(X_test)


helper.MAE(y_test, y_pred, "SVR")


helper.plot(y_test, y_pred, "SVR")



############################################################################

"""
XGBoost
"""

from xgboost import XGBRegressor
# load the dataset

model = XGBRegressor()

chain = RegressorChain(base_estimator=model, order=[0, 1])

start = time.time()

chain.fit(sub_X_train, sub_y_train)

end = time.time()

print("XGBoost training time: " + str(end - start) + "\n")


y_pred = chain.predict(X_test)


helper.MAE(y_test, y_pred, "XGBoost")

helper.plot(y_test, y_pred, "XGBoost")



############################################################################

"""
SGDRegressor
"""

from sklearn.linear_model import SGDRegressor


reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))

chain = RegressorChain(base_estimator=reg, order=[0, 1])

start = time.time()

chain.fit(sub_X_train, sub_y_train)

end = time.time()

print("SGDRegressor training time: " + str(end - start) + "\n")



y_pred = chain.predict(X_test)



helper.MAE(y_test, y_pred, "SGDRegressor")


helper.plot(y_test, y_pred, "SGDRegressor")


quit()




"""
############################################################################

#Linear Regression

from sklearn.linear_model import LinearRegression

start = time.time()

reg = LinearRegression()
setattr(reg, "coef_", (2,20))
reg.fit(features, ground_truth_pair)

end = time.time()

print("LinearRegression")
print(end - start)

y_predict = reg.predict(features[:1000])


t = np.arange(1000)


#plt.plot(t, ground_truth_pair[:1000].T[0], y_predict[:1000].T[0]) #acc

#plt.plot(t, ground_truth_pair[:1000].T[1], y_predict[:1000].T[1]) #vox


#plt.show()
"""
