import os

mingw_path = "C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin"
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

# os.environ['PATH'] = os.environ['PATH'] + ';' +  mingw_path

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

"""
g++ -m64 -std=c++0x -Wall -O3 -msse2  -Wno-unknown-pragmas -funroll-loops -Iinclude -DDMLC_ENABLE_STD_THREAD=0
-Idmlc-core/include -Irabit/include -fopenmp -o xgboost  build/cli_main.o build/learner.o build/logging.o
build/common/hist_util.o build/common/common.o build/c_api/c_api_error.o build/c_api/c_api.o
build/data/simple_dmatrix.o build/data/sparse_page_raw_format.o build/data/data.o build/data/simple_csr_source.o
build/data/sparse_page_writer.o build/data/sparse_page_source.o build/data/sparse_page_dmatrix.o build/gbm/gbm.o
build/gbm/gblinear.o build/gbm/gbtree.o build/metric/multiclass_metric.o build/metric/elementwise_metric.o
build/metric/rank_metric.o build/metric/metric.o build/objective/regression_obj.o build/objective/rank_obj.o
build/objective/objective.o build/objective/multiclass_obj.o build/tree/updater_fast_hist.o build/tree/tree_model.o
build/tree/updater_colmaker.o build/tree/updater_skmaker.o build/tree/updater_sync.o build/tree/updater_refresh.o
build/tree/updater_histmaker.o build/tree/tree_updater.o build/tree/updater_prune.o dmlc-core/libdmlc.a
rabit/lib/librabit_empty.a  -pthread -lm  -fopenmp
"""