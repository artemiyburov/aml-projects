import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing, ensemble
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('task0_sl19d9/train.csv')
df_test = pd.read_csv('task0_sl19d9/test.csv')
df_sample = pd.read_csv('task0_sl19d9/sample.csv')

#print(df_sample)
#regr = linear_model.LinearRegression()
regr = ensemble.RandomForestRegressor()
X = np.array(df_train[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10',]])
#print(X)
y = np.array(df_train['y'])
X_test = np.array(df_test[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10',]])
'''
#preprocessing
X = preprocessing.scale(X)

regr.fit(X, y)

y_pred = regr.predict(X_test) #predicted values without index column

'''
y_pred = np.mean(X_test, axis=1, dtype=np.float64)
#y_pred = np.vstack((np.array(df_test['Id']), y_pred)) #predicted values with index column
df_y_pred = pd.DataFrame({'Id': np.array(df_test['Id']),'y': y_pred})
#print(df_y_pred)
df_y_pred.to_csv('task0_sl19d9/y_pred.csv', index=False)