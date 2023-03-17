# coding:utf-8

import matplotlib.pyplot as plt
import csv


def average(list):
    sumNum = 0
    for i in list:
        sumNum = sumNum + i
    avg = sumNum / len(list)
    return avg


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
    plt.scatter(columnValueKm, columnValuePrice)
    valueKmAverage = average(columnValueKm)
    valuePriceAverage = average(columnValuePrice)
    print("value km average ==>", valueKmAverage)
    print("value price average ==>", valuePriceAverage)
    xtickValue = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
                  180000, 200000, 220000, 240000, 260000, 280000]
    xtickLab = ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k',
                '180k', '200k', '220k', '240k', '260k', '280k']
    yTickValue = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    yTickLab = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']

    plt.xlabel('Number of KM')
    plt.ylabel('Price in K euros')
    plt.title('Price for a car depending on the KM')
    plt.xticks(xtickValue, xtickLab)
    plt.yticks(yTickValue, yTickLab)

    plt.show()
    return 0


if __name__ == "__main__":
    SystemExit(main())
