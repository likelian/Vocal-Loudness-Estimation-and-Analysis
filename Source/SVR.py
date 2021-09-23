from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import matplotlib.pyplot as plt
import os



############################################################################

#initial the model
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

############################################################################

ground_truth_path = "../Ground_truth/"

abs_ground_truth_path = os.path.abspath(ground_truth_path)

for filename in os.listdir(abs_ground_truth_path):
    if filename.endswith(".json"):
        f = open(abs_ground_truth_path+"/"+filename)
        ground_truth_dict = json.load(f)
        for key in ground_truth_dict.keys():
            ground_truth_array = np.array(ground_truth_dict[key])

            #create random feature values
            rng = np.random.RandomState(0)
            X = rng.randn(ground_truth_array.size, 2) #random feature values

            #train the model
            regr.fit(X, ground_truth_array)



############################################################################



y_pred = regr.predict(X)

print(np.array([ground_truth_array.T, y_pred.T]))
t = np.arange(ground_truth_array.size)
plt.plot(t, ground_truth_array, y_pred)
plt.show()


############################################################################
