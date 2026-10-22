import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=False)

    def _rows(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]              # the single amenities column
        elif isinstance(X, np.ndarray):
            X = X.ravel()
        return [list(x) if isinstance(x, (list, tuple, set)) else [x] for x in X]

    def fit(self, X, y=None):
        self.mlb.fit(self._rows(X))
        return self

    def transform(self, X):
        return self.mlb.transform(self._rows(X))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.mlb.classes_)