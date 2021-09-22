from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import matplotlib.pyplot as plt



f = open("/Users/likelian/Desktop/Lab/Lab_fall2021/Ground_truth/ground_truth.json")

ground_truth_dict = json.load(f)


for key in ground_truth_dict.keys():
    ground_truth_array = np.array(ground_truth_dict[key])

############################################################################


rng = np.random.RandomState(0)
X = rng.randn(ground_truth_array.size, 2)

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, ground_truth_array)



############################################################################
