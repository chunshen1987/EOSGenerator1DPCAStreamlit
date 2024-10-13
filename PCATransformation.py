from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCATransformation:
    def __init__(self, explained_var):
        self.n_components = 1
        self.explained_var = explained_var
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.explained_var)

    def fit_transform(self, X):
        X = self.scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        self.n_components = self.pca.n_components_
        self.pcMin = np.min(X, axis=0)
        self.pcMax = np.max(X, axis=0)
        return X

    def inverse_transform(self, X):
        X = self.pca.inverse_transform(X)
        X = self.scaler.inverse_transform(X)
        return X
