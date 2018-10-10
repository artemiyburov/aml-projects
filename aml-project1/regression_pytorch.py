import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from tempfile import mkdtemp
import datetime
from dateutil.parser import parse
import inspect
from numbers import Number
import math

import zlib
import zipfile

class PytorchRegressor(BaseEstimator, RegressorMixin):
    """A pytorch regressor"""

    def __init__(self, output_dim=1, input_dim=100, hidden_layer_dims=[100, 100],
                 num_epochs=1, learning_rate=0.01, batch_size=128, shuffle=False,
                 callbacks=[], use_gpu=True, verbose=1):
        """
        Called when initializing the regressor
        """
        self._history = None
        self._model = None
        self._gpu = use_gpu and torch.cuda.is_available()

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _build_model(self):
        self._layer_dims = [self.input_dim] + \
            self.hidden_layer_dims + [self.output_dim]

        self._model = torch.nn.Sequential()

        # Loop through the layer dimensions and create an input layer, then
        # create each hidden layer with relu activation.
        for idx, dim in enumerate(self._layer_dims):
            if (idx < len(self._layer_dims) - 1):
                module = torch.nn.Linear(dim, self._layer_dims[idx + 1])
                init.xavier_uniform(module.weight)
                self._model.add_module("linear" + str(idx), module)

            if (idx < len(self._layer_dims) - 2):
                self._model.add_module("relu" + str(idx), torch.nn.ReLU())

        if self._gpu:
            self._model = self._model.cuda()

    def _train_model(self, X, y):
        torch_x = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()
        if self._gpu:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train = data_utils.TensorDataset(torch_x, torch_y)
        train_loader = data_utils.DataLoader(train, batch_size=self.batch_size,
                                             shuffle=self.shuffle)

        loss_fn = torch.nn.MSELoss(size_average=False)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate)

        self._history = {"loss": [], "val_loss": [], "mse_loss": []}

        finish = False
        for epoch in range(self.num_epochs):
            if finish:
                break

            loss = None
            idx = 0
            for idx, (minibatch, target) in enumerate(train_loader):
                y_pred = self._model(Variable(minibatch))

                loss = loss_fn(y_pred, Variable(
                    target.cuda().float() if self._gpu else target.float()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_labels = target.cpu().numpy() if self._gpu else target.numpy()
            y_pred_results = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()

            error = mean_absolute_error(y_labels, y_pred_results)

            self._history["mse_loss"].append(loss.data[0])
            self._history["loss"].append(error)

            if self.verbose > 0:
                print("Results for epoch {}, loss {}, mse_loss {}".format(epoch + 1,
                                                                          error, loss.data[0]))
            for callback in self.callbacks:
                callback.call(self._model, self._history)
                if callback.finish:
                    finish = True
                    break

    def fit(self, X, y):
        """
        Trains the pytorch regressor.
        """

        assert (type(self.input_dim) ==
                int), "input_dim parameter must be defined"
        assert (type(self.output_dim) == int), "output_dim must be defined"

        self._build_model()
        self._train_model(X, y)

        return self

    def predict(self, X, y=None):
        """
        Makes a prediction using the trained pytorch model
        """
        if self._history == None:
            raise RuntimeError("Regressor has not been fit")

        results = []
        split_size = math.ceil(len(X) / self.batch_size)

        # In case the requested size of prediction is too large for memory (especially gpu)
        # split into batchs, roughly similar to the original training batch size. Not
        # particularly scientific but should always be small enough.
        for batch in np.array_split(X, split_size):
            x_pred = Variable(torch.from_numpy(batch).float())
            y_pred = self._model(x_pred.cuda() if self._gpu else x_pred)
            y_pred_formatted = y_pred.cpu().data.numpy() if self._gpu else y_pred.data.numpy()
            results = np.append(results, y_pred_formatted)

        return results

    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current implementation
        returns negative mae.
        """
        y_pred = self.predict(X, y)
        return mean_absolute_error(y, y_pred) * -1
#************************************************************************************************************
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, field="logerror", min_val=-0.4, max_val=0.4):
        self.min_val = min_val
        self.max_val = max_val
        self.field = field

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.query(
            "{field} > {min_val} and {field} < {max_val}".format(
                field=self.field,
                min_val=self.min_val,
                max_val=self.max_val))

class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.base_date = datetime.datetime(1600, 1, 1, 0, 0)

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        data["transactiondate"] = data["transactiondate"].apply(
            lambda date: date if isinstance(date, Number) else (
                parse(str(date)) - self.base_date).days)
        return data


class LabelEncodeObjects(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for c in data.columns:
            if data[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(data[c].values))
                data[c] = lbl.transform(list(data[c].values))
        return data


class LabelEncodeCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for col in self.cols:
            lbl = LabelEncoder()
            lbl.fit(list(data[col].values))
            data[col] = lbl.transform(list(data[col].values))
        return data


class NaFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val=-1):
        self.fill_val = fill_val

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.fillna(self.fill_val)


class NaColFiller(BaseEstimator, TransformerMixin):
    def __init(self, fill_val=-1, cols=[]):
        self.fill_val = fill_val
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for col in cols:
            data[col] = data[col].fillna(self.fill_val)
        return data


class NaColMeanFiller(BaseEstimator, TransformerMixin):
    def __init(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for col in cols:
            data[col] = data[col].fillna(data[col].mean)
        return data


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.drop(self.cols, axis=1)


class Cloner(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.copy()


class ColumnOrderer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data.sort_index(axis=1)
#DDDDDDDDDDDDDAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTAAAAAAAAAAAAAAAA
df_X_train = pd.read_csv('task1/X_train.csv')
df_y_train = pd.read_csv('task1/y_train.csv')
df_X_test = pd.read_csv('task1/X_test.csv')

df_train = pd.merge( df_y_train, df_X_train, on='id')
df_train.set_index('id')
df_train.fillna(value=df_train.mean(), inplace=True)
df_X_test.fillna(value=df_X_test.mean(), inplace=True)

def make_train_set():
    reduced = OutlierRemover().transform(df_train)
    x_train = reduced.drop(["logerror"], axis = 1)
    y_train = reduced["logerror"]
    return (x_train, y_train)

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

drop_cols = []
id_cols = ['id']

dp = make_pipeline(Cloner(), DateEncoder(), ColumnDropper(cols=drop_cols),
                   NaFiller(), LabelEncodeObjects(), LabelEncodeCols(cols=id_cols), ColumnOrderer(),
                   StandardScaler())

x_train_df, y_train = make_train_set()
x_train = dp.fit_transform(x_train_df)

learning_rate = 0.0007

clf1 = PytorchRegressor(hidden_layer_dims=[1200, 500, 100, 10],
                        learning_rate=0.0005, num_epochs=10)
clf2 = PytorchRegressor(hidden_layer_dims=[500, 500],
                        learning_rate=learning_rate, num_epochs=10)
clf3 = PytorchRegressor(hidden_layer_dims=[1500, 500, 10],
                        learning_rate=learning_rate, num_epochs=10)
clf4 = PytorchRegressor(hidden_layer_dims=[1000, 800, 500, 200, 100, 10],
                        learning_rate=learning_rate, num_epochs=10)

estimators = [clf1, clf2, clf3, clf4]

for idx, estimator in enumerate(estimators):
    print("Fitting", idx + 1)
    estimator.fit(x_train, y_train.as_matrix())

print("Classifier 1")
print(clf1.predict(x_train))
print("Classifier 2")
print(clf2.predict(x_train))
print("Classifier 3")
print(clf3.predict(x_train))
print("Classifier 4")
print(clf4.predict(x_train))