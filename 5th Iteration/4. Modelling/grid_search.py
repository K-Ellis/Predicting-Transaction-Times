import pandas as pd
import numpy as np
import time
import os  # Used to create folders
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from shutil import copyfile	 # Used to copy parameters file to directory
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from read_parameter_file import get_parameters
# from xgboost import xgboost


parameters = "../../../Data/parameters.txt"	 # Parameters file
d = get_parameters(parameters)


if d["user"] == "Kieron":
    if d["specify_subfolder"] == "n":
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime("%H.%M.%S/")# Log file location
    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + d["specify_subfolder"] + time.strftime("/%Y.%m.%d/") + \
                  time.strftime("%H.%M.%S/")  # Log file location
else:
    newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
if not os.path.exists(newpath):
    os.makedirs(newpath)  # Make folder for storing results if it does not exist

np.random.seed(int(d["seed"]))	# Set seed

if d["user"] == "Kieron":
    df = pd.read_csv(d["file_location"] + d["file_name"] + ".csv", encoding='latin-1', low_memory=False)
else:
    df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["file_name"] + ".csv", encoding='latin-1',
                     low_memory=False)

if d["resample"] == "y":
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

X = df.drop("TimeTaken", axis=1)
y = df["TimeTaken"]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(d["seed"]))


regressors = []
alg_names = []
parameters_to_tune = []

if d["LinearRegression"] == "y":
    regressors.append(LinearRegression())
    parameters_to_tune.append({
        "fit_intercept":[True, False],
        "normalize":[True, False]})
    alg_names.append("LinearRegression")
if d["ElasticNet"] == "y":
    regressors.append(ElasticNet())
    parameters_to_tune.append({
        # "alpha":[0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100], # get convergence warning for small alphas
        "alpha": [0.01, 0.01, 0.1, 1.0, 10, 100],
        "l1_ratio":[.1, .5, .7, .9, .95, .99, 1],
        "max_iter":[10000, 100000],
         # "tol": [0.00001, 0.0001],
         # "warm_start":[True, False]}
    })
    alg_names.append("ElasticNet")
if d["KernelRidge"] == "y":
    regressors.append(KernelRidge(kernel='rbf', gamma=0.1))
    parameters_to_tune.append({"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
    alg_names.append("KernelRidge")
# if d["xgboost"] == "y":
# 	regressors.append(GridSearchCV(xgboost())
# 	alg_names.append("xgboost")
if d["RandomForestRegressor"] == "y":
    regressors.append(RandomForestRegressor())
    parameters_to_tune.append({
        "n_estimators":[100, 250, 500, 1000],
        "criterion":["mse", "mae"],
        "max_features":[1, 0.1, "auto", "sqrt", "log2", None],
        "max_depth":[None, 10, 25, 50]})
    alg_names.append("RandomForestRegressor")


for regressor, alg_name, params in zip(regressors, alg_names, parameters_to_tune):
    print("# Tuning %s hyper-parameters for R-Squared:\n" % (alg_name))#, score))
    regr = GridSearchCV(regressor, params, cv=int(d["crossvalidation"]), scoring=None)
    regr.fit(X_train, y_train.values.ravel())

    print("Best parameters set found on development set:")
    print("\t", regr.best_params_)

    best_train = regr.cv_results_["mean_train_score"][regr.best_index_]
    best_train_std = regr.cv_results_["std_train_score"][regr.best_index_]
    best_test = regr.cv_results_["mean_test_score"][regr.best_index_]
    best_test_std = regr.cv_results_["std_test_score"][regr.best_index_]
    print("\n\t R2 Train: %0.5f (+/- %0.05f)" % (best_train, best_train_std * 2))
    print("\t R2 Test: %0.5f (+/- %0.05f)" % (best_test, best_test_std * 2))

    y_train_pred = regr.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    y_test_pred = regr.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print("\t RMSE Train: %s" % mse_train)
    print("\t RMSE Test: %s" % mse_test)

    with open(newpath + "best_params_%s" % alg_name + time.strftime("%H.%M.%S.txt"), "w") as f:
        f.write("Best parameters set found on development set:\n")
        f.write("\t"+str(regr.best_params_) + "\n")
        f.write("\tTrain: %0.5f (+/- %0.05f)\n" % (best_train, best_train_std * 2))
        f.write("\tTest: %0.5f (+/- %0.05f)\n" % (best_test, best_test_std * 2))
        f.write("\t RMSE Train: %s\n" % mse_train)
        f.write("\t RMSE Test: %s\n" % mse_test)

        print("\nGrid R2 scores on development set:")
        f.write("\nGrid R2 scores on development set:\n")
        means = regr.cv_results_['mean_test_score']
        stds = regr.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, regr.cv_results_['params']):
            f.write("\t%0.5f (+/-%0.05f) for %r\n" % (mean, std * 2, params))
            print("\t%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
