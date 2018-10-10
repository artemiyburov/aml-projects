import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing, svm, ensemble
from sklearn.model_selection import cross_validate #for model selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor

df_X_train = pd.read_csv('task1/X_train.csv')
df_y_train = pd.read_csv('task1/y_train.csv')
df_X_test = pd.read_csv('task1/X_test.csv')
df_X_train.fillna(value=df_X_train.mean(), inplace=True)
df_y_train.fillna(value=df_y_train.mean(), inplace=True)
df_X_test.fillna(value=df_X_test.mean(), inplace=True)

X_train = np.array(df_X_train.drop('id', axis=1))
y_train = np.ravel(np.array(df_y_train.drop('id', axis=1)))
X_test = np.array(df_X_test.drop('id', axis=1))
print(y_train)

#od = ensemble.IsolationForest()
od = LocalOutlierFactor(n_neighbors=int(np.sqrt(y_train.size)))
regr = ensemble.RandomForestRegressor(n_estimators=100)
#regr = linear_model.LinearRegression()
#regr = svm.SVR()
scaler = preprocessing.RobustScaler()

#scaling
scaler.fit_transform(X_train)
scaler.transform(X_test)

#outlier detection
#od.fit(X_train, y_train)
o_train = od.fit_predict(X_train)
print(np.shape(o_train))
oi = np.array([])#array for outlier indices
for i in np.arange(np.shape(o_train)[0]):
    if o_train[i]==-1:
        oi = np.append(oi, i)
        
o_train = np.delete(o_train, oi)
X_train = np.delete(X_train, oi, axis=0)
y_train = np.delete(y_train, oi, axis=0)
print(np.shape(o_train))
print(o_train)

#model selection
k_fold = KFold(n_splits=10) #engineers rule
print(np.mean(cross_val_score(regr, X_train, y_train, cv=k_fold, scoring='r2')))

#fitting the model
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test) #predicted values without index column
df_y_pred = pd.DataFrame({'id': np.array(df_X_test['id']),'y': y_pred})
df_y_pred.to_csv('task1/y_pred.csv', index=False)
