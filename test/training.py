import numpy as np
import matplotlib.pyplot as plt


class FtLinearRegression:
    def __init__(self, file_csv: str, learning_rate: float, iter_number: int):
        self.__grads = np.array([0.0, 0.0])
        self.__data_tab = np.genfromtxt(file_csv, delimiter=",", skip_header=1)
        self.learning_rate = learning_rate
        self.iter_number = iter_number
        self.x = self.__data_tab[:, 0]
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)
        self.y = self.__data_tab[:, 1]
        self.m = len(self.x)
        self.tetha = np.array([0.0, 0.0])
        self.tethas = []
        self.costs = []

    def model(self, x=None):
        if x is None:
            x = self.x
        return x * self.tetha[0] + self.tetha[1]

    def __cost(self, x):
        error = self.model(x) - self.y
        loss = (error**2).sum()
        self.costs.append(loss / (2 * self.m))

    def __gradients(self, x):
        error = self.model(x) - self.y
        self.__grads[0] = (error * x).sum() / self.m
        self.__grads[1] = error.sum() / self.m

    def __gradient_descent(self, x):
        for _ in range(self.iter_number):
            self.__gradients(x)
            self.tetha = self.tetha - self.learning_rate * self.__grads
            self.tethas.append(self.tetha)
            self.__cost(x)

    def get_result(self):
        return self.x, self.y, self.tetha, self.tethas, self.costs

    def standardisation(self):
        return (self.x - self.x_mean) / self.x_std

    def destandardisation(self):
        tetha0 = self.tetha[0] / self.x_std
        tetha1 = self.tetha[1] - (self.tetha[0] * self.x_mean / self.x_std)
        self.tetha = np.array([tetha0, tetha1])

    def train(self):
        x_std = self.standardisation()
        self.__gradient_descent(x_std)
        self.destandardisation()


def main():
    linear_regression = FtLinearRegression("data.csv", 0.005, 1500)
    linear_regression.train()
    x, y, tetha, tethas, costs = linear_regression.get_result()
    plt.figure()
    plt.plot(x, y, "o")
    plt.plot(x, linear_regression.model())
    plt.show()

    plt.figure()
    plt.plot(tethas)
    plt.show()

    plt.figure()
    plt.plot(costs)
    plt.show()

    print("theta ", tetha)
    return 0


if __name__ == "__main__":
    main()
