import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def loss_elem_(y, y_hat) ->np.ndarray:
    return (np.subtract(y, y_hat)** 2)

def loss_(y, y_hat) ->float:
    total = 0
    arrLoss = loss_elem_(y, y_hat)
    for value in arrLoss :
        total = total + value
    size = arrLoss.size
    return float(total / (2 * size))

def loss_elem_square(y, y_hat) ->np.ndarray:
    return (np.subtract(y, y_hat) ** 2)

def mean(y) ->np.ndarray:
    total = 0
    mean = []
    for value in y :
        total = total + value
    size = y.size
    total = total / size
    for i in range(size) :
        mean.append(float(total))
    return (mean)

def loss_elem(y, y_hat) ->np.ndarray:
    return (abs(np.subtract(y, y_hat)))

def mse_(y, y_hat):

    total = 0
    loss = loss_elem_square(y, y_hat)
    for value in loss :
        total = total + value
    size = loss.size
    return (float(total / size))

def rmse_(y, y_hat):
    
    total = 0
    loss = loss_elem_square(y, y_hat)
    for value in loss :
        total = total + value
    size = loss.size
    return (float(sqrt(total / size)))

def mae_(y, y_hat):

    total = 0
    loss = loss_elem(y, y_hat)
    for value in loss :
        total = total + value
    size = loss.size
    return (float(total / size))

def r2score_(y, y_hat):

    loss_num = loss_elem_square(y, y_hat)
    loss_mean = mean(y)
    loss_deno = loss_elem_square(y, loss_mean)
    total1 = 0
    total2 = 0
    for value in loss_num :
        total1 = total1 + value
    for value in loss_deno :
        total2 = total2 + value
    return(float(1 - total1 / total2))


class MyLinearRegression():

    # Description:
    # My personnal linear regression class to fit like a boss.
    # """
    def __init__(self, thetas, alpha=0.005, max_iter=1000000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.thetas_nor0 = 0
        self.thetas_nor1 = 0 

    def predict_(self, x) ->np.ndarray:
        prediction = np.insert(x, 0, 1, axis=1)
        return np.dot(prediction, self.thetas)
    
    def fit_(self, x, y) ->np.ndarray:
        size = y.size
        X_grad = np.insert(x, 0, 1, axis=1)
        for _ in range (self.max_iter) :
            grad = X_grad.T.dot(X_grad.dot(self.thetas) - y) / (2*size)
            self.thetas[0][0] -=  self.alpha * grad[0][0]
            self.thetas[1][0] -=  self.alpha * grad[1][0]
        return self.thetas

#... other methods ...
