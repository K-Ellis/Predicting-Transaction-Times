from sklearn.model_selection import GridSearchCV

def grid_search_CV(regressor, alg_name, parameters_to_tune, newpath, d, X, y):
    print("# Tuning %s hyper-parameters for RMSE:\n" % (alg_name))  # , score))
    if d["grid_search_RMSE"] == "y":
        regr = GridSearchCV(regressor, parameters_to_tune, cv=int(d["crossvalidation"]), scoring="neg_mean_squared_error")
        regr.fit(X, y.values.ravel())

        print("Best parameters found for minimizing RMSE:")
        print("\t", regr.best_params_)

        best_train = np.sqrt(abs(regr.cv_results_["mean_train_score"][regr.best_index_]))
        best_train_std = np.sqrt(regr.cv_results_["std_train_score"][regr.best_index_])
        best_test = np.sqrt(abs(regr.cv_results_["mean_test_score"][regr.best_index_]))
        best_test_std = np.sqrt(regr.cv_results_["std_test_score"][regr.best_index_])
        print("\n\t Train RMSE: %0.1f (+/- %0.1f)" % (best_train, best_train_std * 2))
        print("\t Test RMSE: %0.1f (+/- %0.1f)" % (best_test, best_test_std * 2))

        with open(newpath + "RMSE_best_params_%s" % alg_name, "w") as f:
            f.write("Best parameters found for maximising RMSE:\n")
            f.write("\t" + str(regr.best_params_) + "\n")
            f.write("\tTrain: %0.1f (+/- %0.1f)\n" % (best_train, best_train_std * 2))
            f.write("\tTest: %0.1f (+/- %0.1f)\n" % (best_test, best_test_std * 2))

            print("\nGrid RMSE scores on development set:")
            f.write("\nGrid RMSE scores on development set:\n")
            means = np.sqrt(abs(regr.cv_results_['mean_test_score']))
            stds = np.sqrt(regr.cv_results_['std_test_score'])
            for mean, std, params in zip(means, stds, regr.cv_results_['params']):
                f.write("\t%0.1f (+/- %0.1f) for %r\n" % (mean, std * 2, params))
                print("\t%0.1f (+/- %0.1f) for %r" % (mean, std * 2, params))
    else:
        regr = GridSearchCV(regressor, parameters_to_tune, cv=int(d["crossvalidation"]), scoring=None)
        regr.fit(X, y.values.ravel())

        print("Best parameters found for maximising R2:")
        print("\t", regr.best_params_)

        best_train = regr.cv_results_["mean_train_score"][regr.best_index_]
        best_train_std = regr.cv_results_["std_train_score"][regr.best_index_]
        best_test = regr.cv_results_["mean_test_score"][regr.best_index_]
        best_test_std = regr.cv_results_["std_test_score"][regr.best_index_]
        print("\n\t R2 Train: %0.5f (+/- %0.05f)" % (best_train, best_train_std * 2))
        print("\t R2 Test: %0.5f (+/- %0.05f)" % (best_test, best_test_std * 2))

        with open(newpath + "R2_best_params_%s" % alg_name, "w") as f:
            f.write("Best parameters found for maximising R2:\n")
            f.write("\t" + str(regr.best_params_) + "\n")
            f.write("\tTrain: %0.5f (+/- %0.05f)\n" % (best_train, best_train_std * 2))
            f.write("\tTest: %0.5f (+/- %0.05f)\n" % (best_test, best_test_std * 2))

            print("\nGrid R2 scores on development set:")
            f.write("\nGrid R2 scores on development set:\n")
            means = regr.cv_results_['mean_test_score']
            stds = regr.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, regr.cv_results_['params']):
                f.write("\t%0.5f (+/-%0.05f) for %r\n" % (mean, std * 2, params))
                print("\t%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time
    import os  # Used to create folders
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import mean_squared_error, make_scorer, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from shutil import copyfile  # Used to copy parameters file to directory
    from sklearn.utils import resample
    from read_parameter_file import get_parameters
    # from xgboost import xgboost

    parameters = "../../../Data/parameters.txt"	 # Parameters file
    d = get_parameters(parameters)


    # if d["user"] == "Kieron":
    #     if d["specify_subfolder"] == "n":
    #         newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime("%H.%M.%S/")# Log file location
    #     else:
    #         newpath = r"../0. Results/" + d["user"] + "/model/" + d["specify_subfolder"] + time.strftime("/%Y.%m.%d/") + \
    #                   time.strftime("%H.%M.%S/")  # Log file location

    if d["user"] == "Kieron":
        if d["resample"] == "y":
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] + "/resample/"  # time.strftime(
            # "%Y.%m.%d/") +
            # \time.strftime("%H.%M.%S/")# Log file location
        elif d["specify_subfolder"] == "n":
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"] +"/"#time.strftime("%Y.%m.%d/") +
            # \time.strftime("%H.%M.%S/")# Log file location
        else:
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["input_file"]+ d["specify_subfolder"]+"/" #+
            # time.strftime("/%Y.%m.%d/") + \time.strftime("%H.%M.%S/")  # Log file location


    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    np.random.seed(int(d["seed"]))	# Set seed

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)

    if d["resample"] == "y":
            df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

    X = df.drop("TimeTaken", axis=1)
    y = df["TimeTaken"]

    # Split the dataset in two equal parts
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(d["seed"]))


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
        grid_search_CV(regressor, alg_name,params, newpath, d, X, y)


        # def grid_search_CV(regressor, alg_name, parameters_to_tune, newpath, d, X, y):


    if d["user"] == "Kieron":
        copyfile(parameters, newpath + "parameters.txt")  # Save parameters
    else:
        copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
