import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline


df_X_train = pd.read_csv('../data/raw/X_train.csv')
df_y_train = pd.read_csv('../data/raw/y_train.csv')
df_X_test = pd.read_csv('../data/raw/X_test.csv')

df_train = pd.merge( df_y_train, df_X_train, on='id').set_index('id', drop=True)
#df_train = pd.concat([df_train.loc[df_train['y'] == 1].sample(n=600), df_train.loc[df_train['y'] != 1]]).sample(frac=1).reset_index(drop=True)
print(df_train)
X = df_train
Y = X['y'].values
X = X.drop('y', axis = 1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
scorer = make_scorer(accuracy_score)
'''
pipelines = []
pipelines.append(('ScaledADA', Pipeline([('Scaler', RobustScaler()),('ADA', AdaBoostClassifier())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', RobustScaler()),('GBM', GradientBoostingClassifier())])))
pipelines.append(('ScaledGPR', Pipeline([('Scaler', RobustScaler()),('GPR', RandomForestClassifier())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=42)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scorer)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    '''
scaler = RobustScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
'''param_grid = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":np.arange(210,250)
    }'''
param_grid = {"n_estimators": np.array([60,70,80,90,100,110,120,130,140])}
model = BalancedBaggingClassifier()
kfold = KFold(n_splits=10, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))