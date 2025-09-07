import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def main():
    """Main function"""
    data_tab = np.genfromtxt('data.csv', delimiter=",", skip_header=1)
    data_x = data_tab[:,0]
    data_y = data_tab[:,1]
    plt.plot(data_x, data_y, 'o')
    plt.show()


if __name__ == "__main__":
    main()
