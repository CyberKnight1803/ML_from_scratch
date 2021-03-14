import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FLD():
    def getWeights(self, X_p, X_n):
        """
        Parameters :
        Shape = (N, D) i.e (examples, features)
        
        X_p : Positive examples
        X_n : Negativ Examples

        Return:
        W : Unit Vector
        """

        M_p = np.mean(X_p, axis = 0)
        M_n = np.mean(X_n, axis = 0)

        S1 = np.cov(X_p.T)
        S2 = np.cov(X_n.T)

        Sw = np.add(S1, S2)
        Sw_inv = np.linalg.inv(Sw)

        W = np.dot(Sw_inv, (M_p - M_n))
        W = W / np.linalg.norm(W)

        return W

    def getGaussianCurve(self, X, W):

        """
        Parameters :
        X : Input training examples of a particular class
        W : Unit vector

        Returns :
        x, y, Mean, std, var of gaussian distribution
        """

        X_1D = np.dot(W, X.T)
        mean = np.mean(X_1D)
        std = np.std(X_1D)
        var = np.var(X_1D)

        X_1D = np.sort(X_1D)

        exp = -((X_1D - mean)**2) / (2 * var)
        y = (1 / np.sqrt(2 * np.pi * var)) * np.exp(exp)

        GD = {
            'x': X_1D,
            'y': y,
            'mean': mean,
            'std': std,
            'var': var
        }

        return GD
    
    def getIntersectionPoint(self, X_p, X_n, W):

        """
        Finds the intersection point by solving the Quadratic equation
        
        Returns:
        Intersection point, Guassian Distribution stats of both classes.
        """
        gdP = self.getGaussianCurve(X_p, W)
        gdN = self.getGaussianCurve(X_n, W)

        m_p, v_p = gdP['mean'], gdP['var']
        m_n, v_n = gdN['mean'], gdN['var']

        a = (1 / v_p) - (1 / v_n)
        b = 2 * (m_p / v_p - m_n / v_n)
        c = (((m_n ** 2)/v_n - (m_p ** 2)/v_p) + np.log(v_n/v_p))

        roots = np.roots([a, b, c])
        return roots[1], gdP, gdN

    def predictions(self, X, W, threshold):
        projections = np.dot(W, X.T).reshape(-1, 1)

        P = (projections > threshold).astype(int).reshape(-1, 1)
        return P

    def accuracy(self, y, P):
        acc = np.sum(y == P) / len(y)
        return acc * 100

    def Model(self, X_p, X_n, X, y_p, y_n, y):

        W = self.getWeights(X_p, X_n)
        t, gdP, gdN = self.getIntersectionPoint(X_p, X_n, W)
        
        P = self.predictions(X, W, t)
        acc = self.accuracy(y, P)

        self.modelStats = {
            'weight': W,
            'threshold': t,
            'gdP': gdP,
            'gdN': gdN,
            'predictions': P,
            'acc': acc
        }

        return self.modelStats
    
    def plot1D_projections(self, y):
        P = self.modelStats['predictions']
        threshold = self.modelStats['threshold']
        gdP = self.modelStats['gdP']
        gdN = self.modelStats['gdN']

        points = pd.DataFrame(np.concatenate((P, y)))

        pPoints = points.loc[points[1] == 1][[0]]
        nPoints = points.loc[points[1] == 0][[0]]
    
        plt.plot(pPoints, np.ones(pPoints.shape), '.', color = 'r', label = 'Class 1')
        plt.plot(nPoints, np.ones(nPoints.shape), '.', color = 'b', label = 'Class 0')
        plt.plot([threshold], [1], '.', color = 'black', label = 'Threshold', markersize=10)
        plt.plot([gdP['mean']], [1], 'x', color = 'black', label = "Class 1 mean", markersize = 8)
        plt.plot([gdN['mean']], [1], 'x', color = 'black', label = "Class 0 mean", markersize = 8)
        
        plt.title('Projections onto vector W')
        plt.legend(loc = 'upper right')
        plt.show()

