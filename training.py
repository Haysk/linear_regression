# coding:utf-8

import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
import myLinearRegression as myLR
from myLinearRegression import MyLinearRegression as MyLR
from math import sqrt

def normalize(list) ->list:
    out = list.copy()
    min_value = min(list)
    out -= min_value
    max_value = max(list)
    out /= (max_value - min_value)
    return out

def denormalize(thetas, XKM, YPrice) :
    
    min_km = min(XKM)
    max_km = max(XKM)
    min_price = min(YPrice)
    max_price = max(YPrice)
    delta = np.array([max_km - min_km, max_price - min_price])
    thetas[1] = thetas[1] * delta[1] / delta[0]
    thetas[0] = thetas[0] * delta[1] + min(YPrice) - thetas[1] * min(XKM)
    return thetas

def average(list) ->float:
    sumNum = 0
    for i in list:
        sumNum = sumNum + i
    avg = sumNum / len(list)
    return avg

def deviation(i, average) ->float:
    return i - average

def main() -> int:

    with open('data.csv', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        columnValueKm = []
        columnValuePrice = []
        for row in csv_reader:
            x, y = map(float, row)
            columnValueKm.append(x)
            columnValuePrice.append(y)
    XKm = np.array([columnValueKm]).reshape(-1,1)
    YPrice = np.array([columnValuePrice]).reshape(-1,1)
    linear_model = MyLR(np.array([[-0.0], [0.0]]))
    Y_model = linear_model.predict_(XKm)
    myLR.mse_(YPrice, Y_model)
    mean_squared_error(YPrice, Y_model)

    #Get the lost
    lost = myLR.loss_(YPrice, Y_model)

    #Get Theta
    x_normalize = normalize(XKm)
    y_normalize = normalize(YPrice)
    theta = linear_model.fit_(x_normalize, y_normalize)
    theta = denormalize(theta, XKm, YPrice)
    y_loss = linear_model.predict_(XKm)
    plt.plot(XKm, y_loss)
    plt.scatter(XKm, YPrice)
    # plt.colorbar()

    #Sizing for better look
    xtickValue = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
                  180000, 200000, 220000, 240000, 260000, 280000]
    xtickLab = ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k',
                '180k', '200k', '220k', '240k', '260k', '280k']
    yTickValue = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    yTickLab = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']

    #Label
    plt.xlabel('Number of KM')
    plt.ylabel('Price in K euros')
    plt.title('Price for a car depending on the KM')
    plt.xticks(xtickValue, xtickLab)
    plt.yticks(yTickValue, yTickLab)

    #Write Theta in CSV
    with open('predict.csv', 'w', newline='') as file:
        write = csv.writer(file)
        write.writerows(theta)

    #Show
    plt.show()
    return 0


if __name__ == "__main__":
    SystemExit(main())
