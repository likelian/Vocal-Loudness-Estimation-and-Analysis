
"""
force the subsampling close to a uniform distribution

1. find the mean and var of the ground truth training data y_train
2. generate a random standard distributed array of the given mean and var
3. for each value in the array, find the closest one in y_train
"""
"""
Because accompaniment relative loudness and vocal relative loudness is
highly correlated, we only consider the distribution of accompaniment relative loudness

force the subsampling close to a uniform distribution

1. find the mean and var of the ground truth training data y_train
2. identify the values of mean - var and mean + var (may have a scaling on var)
3. sort the training data y_train
4. randomly (uniform), random order index remove data points within the range of 2.
5. randomly (normal distribution) remove data points from index.
"""

"""

train_stack = np.concatenate([y_train, X_train], axis=1)
sorted_train_stack = train_stack[np.argsort(train_stack[:, 0])]


count, bins, ignored = plt.hist(sorted_train_stack.T[0], 1000, density=True)
#plt.show()
plt.close()

mean = np.mean(y_train.T[0]) #acc
var = np.var(y_train.T[0])
size = y_train.T[0].shape[0]
a = 1
low_bound = mean - a * var
high_bound = mean + a * var

y_train_sorted = np.sort(y_train.T[0])

low_bound_diff_array = np.abs(y_train_sorted - low_bound)
low_bound_index = low_bound_diff_array.argmin()

high_bound_diff_array = np.abs(y_train_sorted - high_bound)
high_bound_index = high_bound_diff_array.argmin()

mean_idx = (high_bound_index + low_bound_index)/2
var_idx = (high_bound_index  - low_bound_index)/2

print(low_bound)
print(low_bound_index)

print(high_bound)
print(high_bound_index)


idx2remove = np.random.normal(mean_idx, var_idx, size*2).astype(int) #1000 is how many data points to remove

idx2remove = idx2remove[(idx2remove < size) & (idx2remove > 0)]
idx2remove = idx2remove[(idx2remove < (mean_idx+2*var_idx)) & (idx2remove > (mean_idx-2*var_idx))]


sorted_train_stack_removed = np.delete(sorted_train_stack, idx2remove, 0)


count, bins, ignored = plt.hist(sorted_train_stack_removed.T[0], 1000, density=True)
#plt.show()
plt.close()


y_train = sorted_train_stack_removed.T[:2].T
X_train = sorted_train_stack_removed.T[2:].T

print(y_train.shape)
print(X_train.shape)


#quit()




"""

"""
Compute the mean and std of the ground truth
"""

"""
mean_individual_loudness = np.mean(y, axis=0)
std_individual_loudness = np.std(y, axis=0)



print("mean_individual_loudness(acc, vox): " + str(mean_individual_loudness))
print("std_individual_loudness(acc, vox): " + str(std_individual_loudness))

"""


############################################################################



"""
Plot ground truth histogram
"""

"""


plot_histogram_ground_truth(y)
"""


############################################################################


############################################################################


############################################################################

"""
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_size = X_train.shape[0]
test_size = X_test.shape[0]

print("train_size")
print(train_size)

print("test_size")
print(test_size)

"""

############################################################################


#subsample

"""
The above data split are ignored
"""
"""

sub_X_train = X_train[0:train_size:5]
sub_y_train = y_train[0:train_size:5]

print("sub_train_size")
print(sub_X_train.shape)
"""

############################################################################


"""
The above data split are ignored

print("The above data split are ignored")
print("split before 1000 and after 1000")
sub_X_train = X[1000:][0:-1:60]
sub_y_train = y[1000:][0:-1:60]
X_test = X[:1000]
y_test = y[:1000]
print("sub_X_train" + str(sub_X_train.shape))
print("y_test" + str(y_test.shape))
"""


############################################################################

"""
Normalization
"""



#from sklearn.preprocessing import MinMaxScaler


"""
#print("MinMaxScaler")
#scaler = MinMaxScaler()

print("StandardScaler")
scaler = StandardScaler()

scaler.fit(sub_X_train)

sub_X_train = scaler.transform(sub_X_train)
X_test = scaler.transform(X_test)
"""


def machine_learning_simple(X, y, file_dict):
    """
    apply machine learning algorithms
    """

    X, y, file_dict

    file_count = len(file_dict.keys())
    error_mean_matrix = np.zeros((file_count, 4))
    error_SVR_matrix = np.zeros((file_count, 4))
    idx = 0

    y_test_mean_total = None
    y_test_SVR_total = None
    y_pred_mean_total = None
    y_pred_SVR_total = None


    print("The above data split are ignored")
    print("split before 1000 and after 1000")
    sub_X_train = X[1000:][0:-1:60]
    sub_y_train = y[1000:][0:-1:60]
    X_test = X[:1000]
    y_test = y[:1000]
    print("sub_X_train" + str(sub_X_train.shape))
    print("y_test" + str(y_test.shape))

    #Normalization
    scaler = StandardScaler()
    scaler.fit(sub_X_train)

    sub_X_train = scaler.transform(sub_X_train)
    X_test = scaler.transform(X_test)

    start = time.time()

    error_mean, y_test_mean, y_pred_mean = Mean_learning(sub_X_train, sub_y_train, X_test, y_test)
    error_SVR, y_test_SVR, y_pred_SVR = SVR_learning(sub_X_train, sub_y_train, X_test, y_test)

    end = time.time()

    error_mean_matrix[idx] = error_mean
    error_SVR_matrix[idx] = error_SVR


    return None

#machine_learning_simple(X, y, file_dict)

quit()

"""
#SVR_chained_vox_first
"""

regr = make_pipeline(SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[1, 0])

start = time.time()

chain.fit(sub_X_train, sub_y_train)

end = time.time()

print("SVR training time: " + str(end - start) + "\n")


y_pred = chain.predict(X_test)


helper.MAE(y_test, y_pred, "SVR_chained_vox_first")
helper.ME(y_test, y_pred, "SVR_chained_vox_first")


helper.plot(y_test, y_pred, "SVR_chained_vox_first")


"""
#SVR_unchained
"""

regr_acc = make_pipeline(SVR(C=1.0, epsilon=0.2))
regr_vox = make_pipeline(SVR(C=1.0, epsilon=0.2))


start = time.time()

regr_acc.fit(sub_X_train, sub_y_train.T[0].T)
regr_vox.fit(sub_X_train, sub_y_train.T[1].T)

end = time.time()


print("SVR training time: " + str(end - start) + "\n")


y_pred_acc = regr_acc.predict(X_test)
y_pred_vox = regr_vox.predict(X_test)


y_pred = np.vstack([y_pred_acc, y_pred_vox]).T


helper.MAE(y_test, y_pred, "SVR_unchained")
helper.ME(y_test, y_pred, "SVR_unchained")


helper.plot(y_test, y_pred, "SVR_unchained")

"""


"""

"""
#Double Chained
"""

#regr_acc = make_pipeline(SVR(C=1.0, epsilon=0.2))
#regr_vox = make_pipeline(SVR(C=1.0, epsilon=0.2))


regr = make_pipeline(SVR(C=1.0, epsilon=0.2))
regr_iter = make_pipeline(SVR(C=1.0, epsilon=0.2))

chain = RegressorChain(base_estimator=regr, order=[0, 1])
chain_iter_1 = RegressorChain(base_estimator=regr, order=[0, 1])



start = time.time()

chain.fit(sub_X_train, sub_y_train)

chain_pred = chain.predict(sub_X_train)

new_sub_X_train = np.concatenate((sub_X_train, chain_pred), axis=1)

chain_iter_1.fit(new_sub_X_train, sub_y_train)


end = time.time()



print("SVR training time: " + str(end - start) + "\n")


y_pred = chain.predict(X_test)
new_X_test = np.concatenate((X_test, y_pred), axis=1)
y_pred_iter_1 = chain_iter_1.predict(new_X_test)


helper.MAE(y_test, y_pred_iter_1, "SVR_chain_iter_1")
helper.ME(y_test, y_pred_iter_1, "SVR_chain_iter_1")


helper.plot(y_test, y_pred_iter_1, "SVR_chain_iter_1")
"""






############################################################################



from xgboost import XGBRegressor
# load the dataset

def XGBRegressor_learning(sub_X_train, sub_y_train, X_test, y_test):
    """
    XGBoost
    """

    model = XGBRegressor()

    chain = RegressorChain(base_estimator=model, order=[0, 1])

    start = time.time()

    chain.fit(sub_X_train, sub_y_train)

    end = time.time()

    print("XGBoost training time: " + str(end - start) + "\n")


    y_pred = chain.predict(X_test)


    helper.MAE(y_test, y_pred, "XGBoost")
    helper.ME(y_test, y_pred, "XGBoost")


    helper.plot(y_test, y_pred, "XGBoost")

    return None

#XGBRegressor_learning(sub_X_train, sub_y_train, X_test, y_test)


############################################################################



from sklearn.linear_model import SGDRegressor


def SGDRegressor_learning(sub_X_train, sub_y_train, X_test, y_test):
    """
    SGDRegressor
    """

    reg = make_pipeline(SGDRegressor(max_iter=10000, tol=1e-3))

    chain = RegressorChain(base_estimator=reg, order=[0, 1])

    start = time.time()

    chain.fit(sub_X_train, sub_y_train)

    end = time.time()

    print("SGDRegressor training time: " + str(end - start) + "\n")



    y_pred = chain.predict(X_test)



    helper.MAE(y_test, y_pred, "SGDRegressor")
    helper.ME(y_test, y_pred, "SGDRegressor")


    helper.plot(y_test, y_pred, "SGDRegressor")

    return None

#SGDRegressor_learning(sub_X_train, sub_y_train, X_test, y_test)

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
