from sklearn.feature_selection import RFECV


def CVRFE(in_regressor, d, X, y, outfile, newpath):
    if d["step"] == "y":
        step = 1 # 1 is the default value
    else:
        step = int(d["step"])

    outfile.write("Recursive Feature Selection Cross Validation (for step = %s): \n" % step)
    
    regressor = RFECV(estimator=in_regressor, step=step, cv=int(d["crossvalidation"]))
    regressor.fit(X, y.values.ravel())

    print("Optimal number of features : %d" % regressor.n_features_)

    y_pred = regressor.predict(X)
    # todo transform predictions



    # number_close_1 = 0  # Use to track number of close estimations within 1 hour
    # number_close_24 = 0  # Use to track number of close estimations within 24 hours
    #
    # mean_time = sum(y.tolist()) / len(y.tolist())  # Calculate mean of
    # # actual
    # std_time = np.std(y.tolist())  # Calculate standard deviation of actual
    # for i in range(len(y_pred)):  # Convert high or low predictions to 0 or 3 std
    #     if y_pred[i] < 0:  # Convert all negative predictions to 0
    #         y_pred[i] = 0
    #     if y_pred[i] > (mean_time + 4*std_time):  # Convert all predictions > 3 std to 3std
    #         y_pred[i] = (mean_time + 4*std_time)
    #     if math.isnan(y_pred[i]):  # If NaN set to 0
    #         y_pred[i] = 0
    #     if abs(y_pred[i] - y.iloc[i, 0]) <= 3600:  # Within 1 hour
    #         number_close_1 += 1
    #     if abs(y_pred[i] - y.iloc[i, 0]) <= 3600 * 24:  # Within 24 hours
    #         number_close_24 += 1
    # number_close_1.append(number_close_1)
    # number_close_24.append(number_close_24)

    # for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
    #     if y_train_pred[i] < 0:  # Convert all negative predictions to 0
    #         y_train_pred[i] = 0
    #     if y_train_pred[i] > (mean_time + 4*std_time):  # Convert all predictions > 3 std to 3std
    #         y_train_pred[i] = (mean_time + 4*std_time)
    #     if math.isnan(y_train_pred[i]):  # If NaN set to 0
    #         y_train_pred[i] = 0
    # for i in range(len(y_test_pred)): # Convert high or low predictions to 0 or 3 std
    #     if y_test_pred[i] < 0:  # Convert all negative predictions to 0
    #         y_test_pred[i] = 0
    #     if y_test_pred[i] > (mean_time + 4*std_time):  # Convert all predictions > 3 std to 3std
    #         y_test_pred[i] = (mean_time + 4*std_time)
    #     if math.isnan(y_test_pred[i]):  # If NaN set to 0
    #         y_test_pred[i] = 0


    print("RMSE = " + str(sqrt(mean_squared_error(y, y_pred))))
    print("R2 = " + str(r2_score(y, y_pred)))

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("\nNumber of features selected")
    plt.ylabel("Cross validation score")# (nb of correct classifications)")
    # plt.ylim(-1,1.1)
    plt.plot(range(1, len(regressor.grid_scores_) + 1), regressor.grid_scores_)
    plt.savefig(newpath + "r2_to_number_features.png")

    supports = regressor.support_
    rankings = regressor.ranking_
    cvX = pd.DataFrame()
    for i, (ranking, support) in enumerate(zip(rankings, supports)):
        print("Used: %s, Ranking: %s, Column name: %s" % (support, ranking, X.columns[i]))
        outfile.write("Used: %s, Ranking: %s, Column name: %s\n" % (support, ranking, X.columns[i]))
        if support == True:
            cvX[X.columns[i]] = X[X.columns[i]]
    print("columns kept = " + str(cvX.columns.tolist()))

    return cvX, regressor


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time
    import os  # Used to create folders
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.metrics import mean_squared_error, make_scorer
    from sklearn.ensemble import RandomForestRegressor
    from shutil import copyfile  # Used to copy parameters file to directory
    from sklearn.utils import resample
    from read_parameter_file import get_parameters
    from sklearn.feature_selection import RFE
    # from xgboost import xgboost
    from sklearn.metrics import mean_squared_error, r2_score
    from math import sqrt
    import matplotlib.pyplot as plt
    import math

    parameters = "../../../Data/parameters.txt"  # Parameters file
    d = get_parameters(parameters)

    if d["user"] == "Kieron":
        if d["specify_subfolder"] == "n":
            newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime(
                "%H.%M.%S/")  # Log file location
        else:
            newpath = r"../0. Results/" + d["user"] + "/model/" + d["specify_subfolder"] + time.strftime("/%Y.%m.%d/") + \
                      time.strftime("%H.%M.%S/")  # Log file location
    else:
        newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

    np.random.seed(int(d["seed"]))  # Set seed

    if d["user"] == "Kieron":
        df = pd.read_csv(d["file_location"] + d["input_file"] + ".csv", encoding='latin-1', low_memory=False)
    else:
        df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["input_file"] + ".csv", encoding='latin-1',
                         low_memory=False)

    if d["resample"] == "y":
        df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

    out_file_name = newpath + time.strftime("%H.%M.%S") + "_" + "RFE_testing" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file


    X = df.drop("TimeTaken", axis=1)
    X = X.drop("HoldDuration", axis=1)
    y = df["TimeTaken"]

    # Split the dataset in two equal parts
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int(d["seed"]))
    # kf = KFold(int(d["crossvalidation"]))
    #
    # for k, (train, test) in enumerate(k_fold.split(X, y)):
    #     lasso_cv.fit(X[train], y[train])
    #     print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
    #           format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))

    # regr, X_train, y_train = RFECV_k_features(LinearRegression(), d, X, y, out_file)
    CVRFE(LinearRegression(), d, X, y, out_file, newpath)

    out_file.close()

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
