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

rand_features = None
rng = np.random.RandomState(0)
############################################################################

for filename in os.listdir(abs_ground_truth_path):

    if filename.endswith(".json"):
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

        if rand_features is None:
            rand_features = rng.randn(length, 30)
        else:
            rand_features = np.concatenate([rand_features, rng.randn(length, 30)], axis=0)

    ground_truth_pair = np.stack((ground_truth_accREL, ground_truth_voxREL), axis=-1)




print("data created")

print(rand_features.shape)
print(ground_truth_accREL.shape)
print(ground_truth_voxREL.shape)
print(ground_truth_pair.shape)




############################################################################

#train the model

from sklearn.linear_model import LinearRegression

start = time.time()

reg = LinearRegression()
setattr(reg, "coef_", (2,30))
reg.fit(rand_features, ground_truth_pair)

end = time.time()

print("LinearRegression")
print(end - start)


y_predict = reg.predict(rng.randn(10, 30))
print(y_predict)


############################################################################

#train the model

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline

from sklearn.multioutput import RegressorChain



reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))

chain = RegressorChain(base_estimator=reg, order=[0, 1])

start = time.time()

chain.fit(rand_features, ground_truth_pair)

end = time.time()

print("SGDRegressor")
print(end - start)





############################################################################

#train the model


from xgboost import XGBRegressor
# load the dataset

model = XGBRegressor()

chain = RegressorChain(base_estimator=model, order=[0, 1])

start = time.time()

chain.fit(rand_features, ground_truth_pair)

end = time.time()

print("XGBRegressor")
print(end - start)

quit()

############################################################################


#initial the model
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[0, 1])

start = time.time()

chain.fit(rand_features, ground_truth_pair)

end = time.time()

print(end - start)


error = mean_squared_error(ground_truth_array_test, y_pred)
print(error)


#print(np.array([ground_truth_array.T, y_pred.T]))
t = np.arange(ground_truth_array_test.size)
plt.plot(t, ground_truth_array_test, y_pred)
plt.show()


############################################################################
