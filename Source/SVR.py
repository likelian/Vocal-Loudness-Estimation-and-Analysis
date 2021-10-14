from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
import time





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

            current_features = np.array([])
            for key in list(feature_dict.keys()):
                #needs to change when we have more feature types
                current_features = np.array(feature_dict[key])


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

"""
print("split before 1000 and after 1000")
sub_X_train = X[1000:]#[0:-1:10]
sub_y_train = y[1000:]#[0:-1:10]
X_test = X[:1000]
y_test = y[:1000]
print(sub_X_train.shape)
"""


############################################################################

"""
Normalization
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(sub_X_train)

sub_X_train = scaler.transform(sub_X_train)
X_test = scaler.transform(X_test)


############################################################################

"""
SVR
"""

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[0, 1])

start = time.time()

chain.fit(sub_X_train, sub_y_train)

end = time.time()

print("SVR training time")
print(end - start)

y_pred = chain.predict(X_test)


#Evaluation

MAE_acc = mean_absolute_error(y_test.T[0].T, y_pred.T[0].T)
MAE_vox = mean_absolute_error(y_test.T[1].T, y_pred.T[1].T)

print("SVR MAE_acc")
print(MAE_acc)

print("SVR MAE_vox")
print(MAE_vox)


#visualization

t = np.arange(y_pred.shape[0])/10

plt.figure()
plt.suptitle("SVR")

plt.subplot(211)  #acc
plt.title('Accompaniment Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[0], label="ground truth")
plt.plot(t, y_pred.T[0], label="predction")
plt.legend(loc='lower center', ncol=2)


plt.subplot(212)  #vox
plt.title('Vocal Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[1], label="ground truth")
plt.plot(t, y_pred.T[1], label="predction")
plt.legend(loc='lower center', ncol=2)

plt.tight_layout(pad=1.0)

plt.show()


quit()


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

print("XGBRegressor")
print(end - start)


y_pred = chain.predict(X_test)



#Evaluation

MAE_acc = mean_absolute_error(y_test.T[0].T, y_pred.T[0].T)
MAE_vox = mean_absolute_error(y_test.T[1].T, y_pred.T[1].T)

print("XGBoost MAE_acc")
print(MAE_acc)

print("XGBoost SVR MAE_vox")
print(MAE_vox)


#visualization

t = np.arange(y_pred.shape[0])/10

plt.figure()
plt.suptitle("XGBoost")

plt.subplot(211)  #acc
plt.title('Accompaniment Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[0], label="ground truth")
plt.plot(t, y_pred.T[0], label="predction")
plt.legend(loc='lower center', ncol=2)


plt.subplot(212)  #vox
plt.title('Vocal Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[1], label="ground truth")
plt.plot(t, y_pred.T[1], label="predction")
plt.legend(loc='lower center', ncol=2)

plt.tight_layout(pad=1.0)

#plt.show()



############################################################################

#SGDRegressor

from sklearn.linear_model import SGDRegressor


reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))

chain = RegressorChain(base_estimator=reg, order=[0, 1])

start = time.time()

chain.fit(sub_X_train, sub_y_train)


end = time.time()

print("SGDRegressor")
print(end - start)

y_pred = chain.predict(X_test)



#Evaluation

MAE_acc = mean_absolute_error(y_test.T[0].T, y_pred.T[0].T)
MAE_vox = mean_absolute_error(y_test.T[1].T, y_pred.T[1].T)

print("SGDRegressor MAE_acc")
print(MAE_acc)

print("SGDRegressor SVR MAE_vox")
print(MAE_vox)


#visualization

t = np.arange(y_pred.shape[0])/10

plt.figure()
plt.suptitle("SGD Linear Regression")

plt.subplot(211)  #acc
plt.title('Accompaniment Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[0], label="ground truth")
plt.plot(t, y_pred.T[0], label="predction")
plt.legend(loc='lower center', ncol=2)


plt.subplot(212)  #vox
plt.title('Vocal Loudness compared to Mixture Loudness')
plt.ylabel('short-term LUFS in dB')
plt.xlabel('time in seconds')
plt.plot(t, y_test.T[1], label="ground truth")
plt.plot(t, y_pred.T[1], label="predction")
plt.legend(loc='lower center', ncol=2)

plt.tight_layout(pad=1.0)

plt.show()


quit()


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
