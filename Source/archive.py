
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
