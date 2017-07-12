print("Starting..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os  # Used to create folders
import math
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from shutil import copyfile	 # Used to copy parameters file to directory
from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from read_parameter_file import get_parameters


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

parameters = {"fit_intercept":["True", "False"],
	"normalize":["True", "False"]
	}

regressor = LinearRegression()
regr = GridSearchCV(regressor, parameters)
regr.fit(X_train, y_train)
print(regr.best_params_)
means = regr.cv_results_['mean_test_score']
stds = regr.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, regr.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print(regr.best_score_)
print(regr.scorer_)



#
# # y_train_pred = regr.predict(y_train)
# # for i in range(len(y_train_pred)):  # Convert high or low predictions to 0 or 3 std
# # 	if y_train_pred[i] < 0:  # Convert all negative predictions to 0
# # 		y_train_pred[i] = 0
# # 	if math.isnan(y_train_pred[i]):  # If NaN set to 0
# # 		y_train_pred[i] = 0
#
# y_test_pred = regr.predict(X_test)
# for i in range(len(y_test_pred)):  # Convert high or low predictions to 0 or 3 std
# 	if y_test_pred[i] < 0:  # Convert all negative predictions to 0
# 		y_test_pred[i] = 0
# 	if math.isnan(y_test_pred[i]):  # If NaN set to 0
# 		y_test_pred[i] = 0
#
# train_rmse = []
# test_rmse = []
# train_r_sq = []
# test_r_sq = []
#
#
# test_rmse.append(sqrt(mean_squared_error(y_train, y_test_pred)))
#
# test_r_sq.append(r2_score(y_train, y_test_pred))
# alg = "Linear Regression"
#
# print(alg + " Test RMSE -> Max: " + str(round(max(test_rmse), 2)) + ", Min: " +
# 	  str(round(min(test_rmse), 2)) + ", Avg: " + str(round(sum(test_rmse) / len(test_rmse), 2)))  # Print RMSE
#
# print(alg + " Test R^2 score -> Max: " + str(round(max(test_r_sq), 2)) + ", Min: " +
# 	  str(round(min(test_r_sq), 2)) + ", Avg: " + str(round(sum(test_r_sq) / len(test_r_sq), 2)))  # Print R Squared
#
#
# print("..Done")
#
#
