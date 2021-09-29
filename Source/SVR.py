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

#initial the model
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

############################################################################



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

ground_truth_array = np.array([])
rand_features = None
rng = np.random.RandomState(0)
############################################################################

for filename in os.listdir(abs_ground_truth_path):

    if filename.endswith(".json"):
        f = open(abs_ground_truth_path+"/"+filename)
        ground_truth_dict = json.load(f)
        for key in list(ground_truth_dict.keys())[-2:]:
            current_ground_truth = np.array(ground_truth_dict[key])
            ground_truth_array = np.concatenate([ground_truth_array, current_ground_truth], axis=None)

            if rand_features is None:
                rand_features = rng.randn(current_ground_truth.size, 30)
            else:
                rand_features = np.concatenate([rand_features, rng.randn(current_ground_truth.size, 30)], axis=0)

print("data created")

print(rand_features.shape)
print(ground_truth_array.shape)

############################################################################

#train the model

rand_features = rand_features
ground_truth_array = ground_truth_array

start = time.time()

regr.fit(rand_features, ground_truth_array)

end = time.time()
print(end - start)


error = mean_squared_error(ground_truth_array_test, y_pred)
print(error)


#print(np.array([ground_truth_array.T, y_pred.T]))
t = np.arange(ground_truth_array_test.size)
plt.plot(t, ground_truth_array_test, y_pred)
plt.show()


############################################################################
