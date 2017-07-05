import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from model import linear_regression, results
import time
import os
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from select_k_importance import select_importants, trim_df, select_top_k_importants

def holdduration_modeling(i, regressor, alg, newpath, scores, pass_number, importances=None):
    i+=1
    # if i%4 == 0:
        # regr = regressor()
    regr = regressor
    regr = regr.fit(trainData_x, trainData_y)
    y_test_pred = regr.predict(testData_x)
    y_train_pred = regr.predict(trainData_x)
    if i%4 == 0:
        importances = regr.feature_importances_

    out_file_name = newpath + time.strftime("%Y%m%d-%H%M%S") + "_" + alg + ".txt"  # Log file name
    out_file = open(out_file_name, "w")  # Open log file
    out_file.write(alg + " " + time.strftime("%Y%m%d-%H%M%S") + "\n\n")

    out_file.write(alg + " Train RMSE: " + str(sqrt(mean_squared_error(trainData_y, y_train_pred))) + "\n")
    out_file.write(alg + " Test RMSE: " + str(sqrt(mean_squared_error(testData_y, y_test_pred))) + "\n\n")
    out_file.write(alg + " Train R^2 scoree: " + str(r2_score(trainData_y, y_train_pred)) + "\n")
    out_file.write(alg + " Test R^2 score: " + str(r2_score(testData_y, y_test_pred)) + "\n")
    print(alg, "Train rmse:", sqrt(mean_squared_error(trainData_y, y_train_pred)))  # Print Root Mean Squared Error
    print(alg, "Test rmse:", sqrt(mean_squared_error(testData_y, y_test_pred)))  # Print Root Mean Squared Error
    print(alg, "Train R^2 score:", r2_score(trainData_y, y_train_pred))  # Print R Squared
    print(alg, "Test R^2 score:", r2_score(testData_y, y_test_pred), "\n")  # Print R Squared

    if pass_number == 1 and i%4==0:
        print("Feature Importances:")
        dfimportances = pd.DataFrame(data=trainData_x.columns, columns=["Columns"])
        dfimportances["importances"] = importances
        dfimportances.to_csv(newpath + "importances.csv", index=False)
        dfimportances = dfimportances.sort_values("importances", ascending=False)
        print(dfimportances[:10], "\n")

        out_file.write("\nFeature Importances:\n")
        for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances["importances"].values.tolist())):
            out_file.write("%d. \"%s\" (%f)\n" % (i, col, importance))

    scores[alg + str(pass_number) + "_RMSE"] = sqrt(mean_squared_error(testData_y, y_test_pred))
    scores[alg + str(pass_number) + "_R2"] = r2_score(testData_y, y_test_pred)

    out_file.close()

parameters = "../../../Data/parameters.txt"  # Parameters file

d = {}
with open(parameters, "r") as f:
    for line in f:
        line = line.replace(":", "")
        (key, val) = line.split()
        d[key] = val

np.random.seed(int(d["seed"]))  # Set seed

if d["user"] == "Eoin":
    newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist
else:
    newpath = r"../0. Results/" + d["user"] + "/model/" + time.strftime("%Y.%m.%d/") + time.strftime(
        "%H.%M.%S/")  # Log file location
    if not os.path.exists(newpath):
        os.makedirs(newpath)  # Make folder for storing results if it does not exist

if d["user"] == "Eoin": df = pd.read_csv(d["file_location"] + "vw_Incident_cleaned" + d["file_name"] + ".csv",
                                         encoding='latin-1', low_memory=False)
else: df = pd.read_csv(d["file_location"] + d["file_name"] + ".csv", encoding='latin-1', low_memory=False)

if d["resample"] == "y": df = resample(df, n_samples=int(d["n_samples"]), random_state=int(d["seed"]))

X = df.drop("HoldDuration", axis=1)
y = df["HoldDuration"]

trainData_x, testData_x, trainData_y, testData_y = train_test_split(X, y)

regressors = [LinearRegression(), ElasticNet(), KernelRidge(), RandomForestRegressor(n_estimators = int(d[
                                                                                                         "n_estimators"]))]
algs = ["LinearRegression", "ElasticNet", "KernelRidge", "RandomForestRegressor"]
# params = [None, None, None, [n_estimators=int(d["n_estimators"])]]
scores = {}

for i, (regressor, alg) in enumerate(zip(regressors, algs)):
    holdduration_modeling(i, regressor, alg, newpath, scores, pass_number=1)
# cols_to_be_deleted = select_importants(newpath + "importances.csv", thresh=0.001) # keep above threshold

k = 30
cols_to_be_deleted = select_top_k_importants(newpath + "importances.csv", k) # keep top k
df2 = trim_df(df, cols_to_be_deleted)
with open(newpath + "cols_deleted_k=%s_" % k + time.strftime("%H.%M.%S.txt"), "w") as f:
    f.write(str(cols_to_be_deleted))

X = df2.drop("HoldDuration", axis=1)
y = df2["HoldDuration"]

trainData_x, testData_x, trainData_y, testData_y = train_test_split(X, y)


for i, (regressor, alg) in enumerate(zip(regressors, algs)):
    holdduration_modeling(i, regressor, alg, newpath, scores, pass_number=2)

best_RMSE = None
best_R2 = None

pass_numbers = [1, 2]
for alg in algs:
    for pass_number in pass_numbers:
        print("%s RMSE = " % (alg + str(pass_number)), scores[alg + str(pass_number) + "_RMSE"])
        print("%s R2 = " % (alg + str(pass_number)), scores[alg + str(pass_number) + "_R2"])
        if best_RMSE is None:
            best_RMSE = [alg + str(pass_number), scores[alg + str(pass_number) + "_RMSE"]]
            best_R2 = [alg + str(pass_number), scores[alg + str(pass_number) + "_R2"]]
        else:
            if scores[alg + str(pass_number) + "_RMSE"] < best_RMSE[1]:
                best_RMSE = [alg + str(pass_number), scores[alg + str(pass_number) + "_RMSE"]]
            if scores[alg + str(pass_number) + "_R2"] < best_R2[1]:
                best_R2 = [alg + str(pass_number), scores[alg + str(pass_number) + "_R2"]]
print("\nbest_RMSE", best_RMSE)
print("best_R2", best_R2)

# TODO - take the model with the best scores and classify the entire Incident dataset and export it as a new CSV file