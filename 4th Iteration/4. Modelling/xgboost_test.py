import xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

df = pd.DataFrame({'X': [3, 4, 6], 'y': [12, 21, 87]})
X = df.drop('y', axis=1)
y = df['y']

params = {
	'max_depth': 5,
	'n_estimators': 50,
	'objective': 'reg:linear'}

model = xgboost.XGBRegressor(**params)
model.fit(X, y)

x_test = pd.DataFrame({'X': [4, 1, 7]})
pred = model.predict(x_test)