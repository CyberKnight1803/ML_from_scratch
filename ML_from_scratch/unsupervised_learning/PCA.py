import numpy as np 

class PCA():
    def __init__(self, new_dimension):
        self.n_components = new_dimension
    
    def standardize(self, X):
        """
        X.shape = (n_samples, features)
        """
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X

    def transform(self, X):
        X = self.standardize(X)
        covariance_matrix = np.cov(X.T)

        # eigenvector[:, x] corrosponds to eigenvalue[x]
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Take top n components
        args = eigenvalues.argsort()[::-1]
        top_eigenvalues = eigenvalues[args][:self.n_components] 
        top_eigenvectors = np.atleast_1d(eigenvectors[:, args])[:, self.n_components]

        return np.dot(X, top_eigenvectors)

