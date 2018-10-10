import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


df_X_train = pd.read_csv('task1/X_train.csv')
df_y_train = pd.read_csv('task1/y_train.csv')
df_X_test = pd.read_csv('task1/X_test.csv')

df_train = pd.merge( df_y_train, df_X_train, on='id')
df_train.set_index('id')
df_train.fillna(value=df_train.mean(), inplace=True)
df_X_test.fillna(value=df_X_test.mean(), inplace=True)
corr_matrix = df_train.corr(method='pearson').abs()
#print(df_train[columns].describe())
'''
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))# Select upper triangle of correlation matrix
to_drop = [column for column in upper.columns if (any(upper[column] > 0.75) and upper.iloc[1][column] < 0.4)]# Find index of feature columns with correlation greater than 0.95
print(df_train)
df_train = df_train.drop(to_drop, axis=1)
df_X_test = df_X_test.drop(to_drop, axis = 1)
'''
#columns = corr_matrix.nlargest(500, 'y').index
#print(df_train[columns].describe())
#df_train = df_train[columns]
'''
#plotting
correlation_map = np.corrcoef(df_train[columns].values.T)
#sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()'''


X = df_train
Y = X['y'].values
X = X.drop('y', axis = 1).values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
'''
pipelines = []
#pipelines.append(('ScaledLR', Pipeline([('Scaler', RobustScaler()),('LR',LinearRegression())])))
#pipelines.append(('ScaledLASSO', Pipeline([('Scaler', RobustScaler()),('LASSO', Lasso())])))
#pipelines.append(('ScaledEN', Pipeline([('Scaler', RobustScaler()),('EN', ElasticNet())])))
#pipelines.append(('ScaledKNN', Pipeline([('Scaler', RobustScaler()),('KNN', KNeighborsRegressor())])))
#pipelines.append(('ScaledCART', Pipeline([('Scaler', RobustScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', RobustScaler()),('GBM', GradientBoostingRegressor())])))
#pipelines.append(('ScaledGPR', Pipeline([('Scaler', RobustScaler()),('GPR', GaussianProcessRegressor())])))
#pipelines.append(('ScaledRFR', Pipeline([('Scaler', RobustScaler()),('RFR', RandomForestRegressor())])))
#pipelines.append(('ScaledADA', Pipeline([('Scaler', RobustScaler()),('ADA', AdaBoostRegressor())])))
#pipelines.append(('ScaledETR', Pipeline([('Scaler', RobustScaler()),('ETR', ExtraTreesRegressor())])))
pipelines.append(('ScaledMLPR', Pipeline([('Scaler', RobustScaler()),('MLPR', MLPRegressor(solver='lbfgs', hidden_layer_sizes=[100], max_iter=2000, activation='logistic'))])))


results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=42)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='r2')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
'''
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
#param_grid = dict(n_estimators=np.array([76,77,78]))
param_grid = dict(LOF__n_neighbors=np.arange(1,200))
model = Pipeline([('GBM', GradientBoostingRegressor())])
kfold = KFold(n_splits=10, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
