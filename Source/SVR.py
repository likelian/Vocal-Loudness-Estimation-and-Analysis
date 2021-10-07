from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import time


############################################################################
"""
f = open("/Users/likelian/Desktop/Lab/Lab_fall2021/Test/abjones_1_ground_truth.json")
ground_truth_dict = json.load(f)
for key in ground_truth_dict.keys():
    ground_truth_array_test = np.array(ground_truth_dict[key])
rng = np.random.RandomState(0)
rand_features_test = rng.randn(ground_truth_array_test.size, 30)
rand_features_test = np.expand_dims(ground_truth_array_test, axis=1)
"""
############################################################################



ground_truth_path = "../Ground_truth/"
abs_ground_truth_path = os.path.abspath(ground_truth_path)


ground_truth_voxREL = np.array([])
ground_truth_accREL = np.array([])


feature_path = "../Features/"
abs_feature_path = os.path.abspath(feature_path)


features = None

############################################################################

rand_features = None
rng = np.random.RandomState(0)
############################################################################


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


        if rand_features is None:
            rand_features = rng.randn(length, 30)
        else:
            rand_features = np.concatenate([rand_features, rng.randn(length, 30)], axis=0)


    ground_truth_pair = np.stack((ground_truth_accREL, ground_truth_voxREL), axis=-1)


############################################################################




print("data created")
print(unmatched)
print(rand_features.shape)
print(ground_truth_accREL.shape)
print(ground_truth_voxREL.shape)
print(ground_truth_pair.shape)
print(features.shape)





############################################################################

#train the model

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

#acc
#plt.plot(t, ground_truth_pair[:1000].T[0], y_predict[:1000].T[0])

#vox
#plt.plot(t, ground_truth_pair[:1000].T[1], y_predict[:1000].T[1])
#plt.show()










############################################################################

#train the model

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

from sklearn.multioutput import RegressorChain



reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))

chain = RegressorChain(base_estimator=reg, order=[0, 1])

start = time.time()

chain.fit(features, ground_truth_pair)



end = time.time()

print("SGDRegressor")
print(end - start)

y_predict = chain.predict(features[:1000])

t = np.arange(1000)


plt.plot(t, ground_truth_pair[:1000].T[0], y_predict[:1000].T[0]) #acc

#plt.plot(t, ground_truth_pair[:1000].T[1], y_predict[:1000].T[1]) #vox

plt.show()



############################################################################

#train the model


from xgboost import XGBRegressor
# load the dataset

model = XGBRegressor()

chain = RegressorChain(base_estimator=model, order=[0, 1])

start = time.time()

chain.fit(features, ground_truth_pair)

end = time.time()

print("XGBRegressor")
print(end - start)

y_predict = chain.predict(features[:1000])

t = np.arange(1000)

plt.plot(t, ground_truth_pair[:1000].T[0], y_predict[:1000].T[0]) #acc


#plt.plot(t, ground_truth_pair[:1000].T[1], y_predict[:1000].T[1]) #vox

plt.show()


############################################################################


#initial the model

"""
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[0, 1])

start = time.time()

chain.fit(features, ground_truth_pair)

end = time.time()

print(end - start)
"""




#error = mean_squared_error(ground_truth_array_test, y_pred)
#print(error)




y_predict = chain.predict(features[:1000])

t = np.arange(1000)

plt.plot(t, ground_truth_pair[:1000].T[0], y_predict[:1000].T[0]) #acc


#plt.plot(t, ground_truth_pair[:1000].T[1], y_predict[:1000].T[1]) #vox

plt.show()





############################################################################
