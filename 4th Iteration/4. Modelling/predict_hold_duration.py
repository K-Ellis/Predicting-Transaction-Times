import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from shutil import copyfile  # Used to copy parameters file to directory
import time
import os
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from select_k_importance import select_importants, trim_df, select_top_k_importants
from sklearn import preprocessing

# infile needs to be the Incident cleaned data plus the summed HoldDuration column from HoldActivity

# find the alg with the best R2 to predict HoldDurations given the incident worksheet info

def holdduration_modeling_passes(trainData_x, testData_x, trainData_y, testData_y,regressor, alg, newpath, scores,
                                 pass_number):
    regr = regressor
    regr = regr.fit(trainData_x, trainData_y)
    y_test_pred = regr.predict(testData_x)
    y_train_pred = regr.predict(trainData_x)
    if alg == "RandomForestRegressor":
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

    if pass_number is not None and alg == "RandomForestRegressor":
        print("Feature Importances:")
        dfimportances = pd.DataFrame(data=trainData_x.columns, columns=["Columns"])
        dfimportances["importances"] = importances
        dfimportances.to_csv(newpath + "importances_%s.csv" % pass_number, index=False)
        dfimportances = dfimportances.sort_values("importances", ascending=False)
        print(dfimportances[:10], "\n")

        out_file.write("\nFeature Importances:\n")
        for i, (col, importance) in enumerate(zip(dfimportances["Columns"].values.tolist(), dfimportances["importances"].values.tolist())):
            out_file.write("%d. \"%s\" (%f)\n" % (i, col, importance))
    if pass_number is not None:
        scores[alg + str(pass_number) + "_RMSE"] = sqrt(mean_squared_error(testData_y, y_test_pred))
        scores[alg + str(pass_number) + "_R2"] = r2_score(testData_y, y_test_pred)
        scores[alg + str(pass_number)] = regr
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
X = X.drop("TimeTaken", axis=1)
y = df["HoldDuration"]

trainData_x, testData_x, trainData_y, testData_y = train_test_split(X, y)


regressors = []
algs = []
if d["linear_regression"] == "y":
    regressors.append(LinearRegression())
    algs.append("LinearRegression")
if d["elastic_net"] == "y":
    regressors.append(ElasticNet())
    algs.append("ElasticNet")
if d["kernel_ridge"] == "y":
    regressors.append(KernelRidge())
    algs.append("KernelRidge")
if d["random_forest_regressor"] == "y":
    regressors.append(RandomForestRegressor(n_estimators = int(d["n_estimators"])))
    algs.append("RandomForestRegressor")

scores = {}
for regressor, alg in zip(regressors, algs):
    holdduration_modeling_passes(trainData_x, testData_x, trainData_y, testData_y,regressor, alg, newpath, scores,
                                 pass_number=1)

# keep the top 30 columns and predict again
k = 30
cols_to_be_deleted = select_top_k_importants(newpath + "importances_1.csv", k)  # keep top k
df2 = trim_df(df, cols_to_be_deleted)
with open(newpath + "cols_deleted_k=%s_" % k + time.strftime("%H.%M.%S.txt"), "w") as f:
    f.write(str(cols_to_be_deleted))

X = df2.drop("HoldDuration", axis=1)
X = X.drop("TimeTaken", axis=1)
y = df2["HoldDuration"]

trainData_x, testData_x, trainData_y, testData_y = train_test_split(X, y)

for regressor, alg in zip(regressors, algs):
    holdduration_modeling_passes(trainData_x, testData_x, trainData_y, testData_y, regressor, alg, newpath, scores,
                                 pass_number=2)


# find the alg_names with the best R2
best_RMSE = []
best_R2 = []
pass_numbers = [1, 2]
for alg in algs:
    for pass_number in pass_numbers:
        # print("%s RMSE = " % (alg + str(pass_number)), scores[alg + str(pass_number) + "_RMSE"])
        # print("%s R2 = " % (alg + str(pass_number)), scores[alg + str(pass_number) + "_R2"])
        best_RMSE.append((alg + str(pass_number), scores[alg + str(pass_number) + "_RMSE"]))
        best_R2.append((alg + str(pass_number), scores[alg + str(pass_number) + "_R2"]))

        # if best_RMSE is None:
        #     best_RMSE = [alg + str(pass_number), scores[alg + str(pass_number) + "_RMSE"]]
        #     best_R2 = [alg + str(pass_number), scores[alg + str(pass_number) + "_R2"]]
        # else:
        #     if scores[alg + str(pass_number) + "_RMSE"] < best_RMSE[1]:
        #         best_RMSE = [alg + str(pass_number), scores[alg + str(pass_number) + "_RMSE"]]
        #     if scores[alg + str(pass_number) + "_R2"] < best_R2[1]:
        #         best_R2 = [alg + str(pass_number), scores[alg + str(pass_number) + "_R2"]]
best_RMSE.sort(key=lambda x: x[1])
best_R2.sort(key=lambda x: x[1], reverse=True)
print("\nbest_RMSE list", best_RMSE)
print("best_R2 list", best_R2, "\n")

out_file_name = newpath + time.strftime("%Y%m%d-%H%M%S") + "_final_predictions.txt"  # Log file name
out_file = open(out_file_name, "w")  # Open log file
out_file.write("best_RMSE list: " + str(best_RMSE))
out_file.write("\nbest_R2 list: " + str(best_R2) + "\n")

# use the alg with the best R2 to predict the hold duration given the incident worksheet
for alg, regressor in zip(algs, regressors):
    for i in range(1,3):
        if best_R2[0][0] == alg + str(i):
            regr = scores[alg + str(i)]
            print("Regressor to be used:")
            print(regr)
            if i == 1:
            #     df3 = trim_df(df, cols_to_be_deleted)
            #     X = df3.drop("HoldDuration", axis=1)
            #     X = X.drop("TimeTaken", axis=1)
            #     y = df3["HoldDuration"]
            # else:
                X = df.drop("HoldDuration", axis=1)
                X = X.drop("TimeTaken", axis=1)
                y = df["HoldDuration"]
regr = regr.fit(X, y)
hold_predictions = regr.predict(X)
min_max_scaler = preprocessing.MinMaxScaler()

# scale predictions
hold_predictions_scaled = pd.DataFrame(min_max_scaler.fit_transform(hold_predictions.reshape(-1, 1)))

# return the final df (incident + predicted HoldDurations)
finaldf = df.copy()
del finaldf["HoldDuration"]
finaldf["HoldDuration"] = hold_predictions_scaled

out_file.write("Final Train RMSE: " + str(sqrt(mean_squared_error(y, hold_predictions))) + "\n")
out_file.write("Final Train R^2 scoree: " + str(r2_score(y, hold_predictions)) + "\n")
copyfile(parameters, newpath + "/" + time.strftime("%H.%M.%S") + "_parameters.txt")  # Save parameters

print("\nfinal", "Train rmse:", sqrt(mean_squared_error(y, hold_predictions)))  # Print Root Mean Squared Error
print("final", "Train R^2 score:", r2_score(y, hold_predictions))  # Print R Squared

finaldf.to_csv(d["file_location"] + "vw_Incident_cleaned(HoldDuration_predictions).csv", index=False)