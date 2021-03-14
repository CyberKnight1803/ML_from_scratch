import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate, n_iters):

        self.lRate = learning_rate
        self.n_iters = n_iters

    
    def initializeParams(self, X):
        """
        Argument:
        X : Input vector having shape (n_x, M)

        Initializes params : 
        W : (n_x, 1) 
        b : 0  will broadcast to (1, M) 
        """

        self.n_x, self.M = X.shape
        
        self.W = np.random.randn(self.n_x, 1) * 0.01
        self.b = 0

    def sigmoid(self, Z):
        """
        Calculates the sigmoid function
        """
        
        S = 1 / (1 + np.exp(-Z))
        return S

    def forward_propogation(self, X):
        """
        Arguments :
        X : Inpute vector of shape (n_x, M)

        Returns:
        A : y hat
        """

        Z = np.dot(self.W.T, X) + self.b
        A = self.sigmoid(Z)

        cache = {"A": A, "Z": Z}

        return A, cache
    
    def compute_cost(self, A, y):
        """
        Arguments :
        A : y hat
        y : target value of trainging examples

        Returns :
        J : Total entropy cost function
        """

        J = np.multiply(y, np.log(A)) + np.multiply(1 - y, np.log(1 - A))
        J = np.squeeze(-np.sum(J) / self.M)
        return J
    
    def backward_propagation(self, cache, X, y):
        """
        Arguments :
        cache : Sotres A, Z of previous iterations
        X : Input vector
        y : Target value

        Returns:
        Updates parameters with the help of gradient descent
        grads : dW & db {Derivatives of cost function wrt to dW and db}
        """
        A = cache["A"]
        dZ = A - y
        dW = np.dot(X, dZ.T)/ self.M
        db = np.sum(dZ, axis= 1, keepdims= True)/ self.M

        #Gradient Descent updates
        self.W -= self.lRate * dW
        self.b -= self.lRate * db

        grads = {"dW": dW, "db": db}

        return grads

    def fit(self, X, y, print_cost = False):
        """
        Fitting the entire model with the help of this function.
        """
        
        self.initializeParams(X)
        
        for i in range(0, self.n_iters):
            
            A, cache = self.forward_propogation(X)
            totalCost = self.compute_cost(A, y)
            grads = self.backward_propagation(cache, X, y)

            if print_cost and i % 200 == 0:
                print(f"Cost after iteration {i} : {totalCost}")

    def predict(self, X):

        A, cache = self.forward_propogation(X)
        predictions = np.around(A)

        return predictions

    def accuracy(self, X, y):

        predictions = self.predict(X)
        acc = np.sum(predictions == y) / y.shape[1]
        return acc * 100



    


