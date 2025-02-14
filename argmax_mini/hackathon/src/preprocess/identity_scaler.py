import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array(X)

    def inverse_transform(self, X):
        return np.array(X)
