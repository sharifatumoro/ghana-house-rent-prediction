import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer(sparse_output=False)

    def fit(self, X, y=None):
        # Ensure each row is a list
        X = [x if isinstance(x, (list, tuple, set)) else [x] for x in X]
        self.mlb.fit(X)
        return self

    def transform(self, X):
        X = [x if isinstance(x, (list, tuple, set)) else [x] for x in X]
        return self.mlb.transform(X)

    def get_feature_names_out(self, input_features=None):
        # Return raw amenity names (no prefix)
        return np.array(self.mlb.classes_)
