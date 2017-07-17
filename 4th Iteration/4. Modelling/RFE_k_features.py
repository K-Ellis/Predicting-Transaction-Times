from sklearn.feature_selection import RFE


def RFE_k_features(in_regressor, d, X_train, y_train, X_test, y_test, outfile):
    if d["top_k_features"] == "y":
        k = None # when k is None, it keeps the top 50% of features
    else:
        k = int(d["top_k_features"])
    if d["step"] == "y":
        step = 1 # 1 is the default value
    else:
        step = int(d["step"])

    outfile.write("Recursive Feature Selection: \n")
    outfile.write("\t k = " + str(k) + "\n")
    outfile.write("\t step = " + str(step) + "\n\n")
    
    original_features_regr = RFE(in_regressor, n_features_to_select=k, step=step)
    original_features_regr.fit(X_train, y_train.values.ravel())
    
    print("train score (all features): ", original_features_regr.score(X_train, y_train))
    print("test score (all features): ", original_features_regr.score(X_test, y_test), "\n")
    outfile.write("train score (all features): " + str(original_features_regr.score(X_train, y_train)) + "\n")
    outfile.write("test score (all features): " + str(original_features_regr.score(X_test, y_test)) + "\n\n")
    
    supports = original_features_regr.support_
    rankings = original_features_regr.ranking_
    
    kX_train = pd.DataFrame()
    kX_test = pd.DataFrame()
    for i, (ranking, support) in enumerate(zip(rankings, supports)):
        print("Used: %s, Ranking: %s, Column name: %s" % (support, ranking, X_train.columns[i]))
        outfile.write("Used: %s, Ranking: %s, Column name: %s\n" % (support, ranking, X_train.columns[i]))
        if support == True:
            kX_train[X_train.columns[i]] = X_train[X_train.columns[i]]
            kX_test[X_train.columns[i]] = X_test[X_train.columns[i]]

    RFE_features_regr = in_regressor
    RFE_features_regr.fit(kX_train, y_train.values.ravel())

    if k is None:
        print("\ntrain score (k=half_features and step=%s): " % step, RFE_features_regr.score(kX_train, y_train))
        print("test score: (k=half_features and step=%s): " % step, RFE_features_regr.score(kX_test, y_test))

        outfile.write("\n\ntrain score (k=half_features and step=%s): " % step + str(RFE_features_regr.score(kX_train,
                                                                                                      y_train)))
        outfile.write("\ntest score: (k=half_features and step=%s): " % step + str(RFE_features_regr.score(kX_test,
                                                                                                        y_test)))

    else:
        print("\ntrain score (k=%s_features and step=%s): " % (k, step) + str(RFE_features_regr.score(kX_train,
                                                                                                        y_train)))
        print("test score: (k=%s_features and step=%s): " % (k, step) + str(RFE_features_regr.score(kX_test, y_test)))

        outfile.write("\n\ntrain score (k=%s_features and step=%s): " % (k, step), RFE_features_regr.score(kX_train,
                                                                                                      y_train))
        outfile.write("\ntest score: (k=%s_features and step=%s): " % (k, step), RFE_features_regr.score(kX_test,
                                                                                                           y_test))

    if RFE_features_regr.score(kX_test, y_test) > original_features_regr.score(X_test, y_test):
        print("\n\nUsing RFE features: ")
        print("\tRFE Features Test Score > Original Features Test Score")
        print("\t%s > %s" % (RFE_features_regr.score(kX_test, y_test), original_features_regr.score(X_test, y_test)))
        out_file.write("\n\nUsing RFE features\n")
        out_file.write("\tRFE Features Test Score > Original Features Test Score\n")
        out_file.write("\t%s > %s\n" % (RFE_features_regr.score(kX_test, y_test), original_features_regr.score(X_test, y_test)))
        return RFE_features_regr, kX_train, kX_test
    else:
        print("\n\nUsing Original Features: ")
        print("\tRFE Features Test Score < Original Features Test Score")
        print("\t%s < %s" % (RFE_features_regr.score(kX_test, y_test), original_features_regr.score(X_test, y_test)))
        out_file.write("\n\nUsing Original Features\n")
        out_file.write("\tRFE Features Test Score < Original Features Test Score\n")
        out_file.write("\t%s < %s\n" % (RFE_features_regr.score(kX_test, y_test), original_features_regr.score(X_test, y_test)))
        return original_features_regr, X_train, X_test
    
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

    out_file_name = newpath + time.strftime("%H.%M.%S") + "_" + "RFE_testing" + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file

    regr, X_train, y_train = RFE_k_features(LinearRegression(), d, X_train, y_train, X_test, y_test, out_file)

    out_file.close()

    copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters
