# coding:utf-8

import csv
import numpy as np
import matplotlib.pyplot as plt

def value_Price(val, theta) ->np.ndarray:

    out = []
    out = val * theta[1][0] + theta[0][0]
    return out

def line(x, theta):
    return x * (theta[1][0]) + (theta[0][0])

def main() ->int:

    try:
        theta = []
        y = []
        y_predict = []
        x = [10000, 40000, 60000, 80000, 400000]
        theta = np.genfromtxt('predict.csv', delimiter='')
        val = input("For how many Km do you want to evaluate the value of the car? ")
        val = np.array(float(val))
        x = np.array((x))
        theta = theta.reshape(-1, 1)
        for value in x :
            y.append(line(value, theta))
        y_predict = value_Price(val, theta)
        if (y_predict < 0) :
            y_predict = 0
        print("The estimited price for the car you ask is", y_predict)
        plt.plot(x, y)
        plt.scatter(val, y_predict)
        plt.show()

    except FileNotFoundError:
        print("Value NULL")
        print('You need to train the model first')
    return 0

if __name__ == "__main__":
    SystemExit(main())