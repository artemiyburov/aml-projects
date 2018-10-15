import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline


df_X_train = pd.read_csv('../data/raw/X_train.csv')
df_y_train = pd.read_csv('../data/raw/y_train.csv')
df_X_test = pd.read_csv('../data/raw/X_test.csv').set_index('id', drop=True)

df_train = pd.merge( df_y_train, df_X_train, on='id').set_index('id', drop=True)
#df_train = pd.concat([df_train.loc[df_train['y'] == 1].sample(n=600), df_train.loc[df_train['y'] != 1]]).sample(frac=1).reset_index(drop=True)
print(df_train)
X = df_train
Y = X['y'].values
X = X.drop('y', axis = 1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=42)
scorer = make_scorer(accuracy_score)
X_train = X
y_train = Y
X_test = df_X_test.values

scaler = RobustScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)
rescaled_X_test = scaler.transform(X_test)
model = BalancedRandomForestClassifier(random_state=42, n_estimators=156)
model.fit(rescaled_X_train, y_train)
y_pred = model.predict(rescaled_X_test) #predicted values without index column
df_y_pred = pd.DataFrame({'id': np.arange(np.size(y_pred)),'y': y_pred})
df_y_pred.to_csv('../data/processed/y_pred.csv', index=False)