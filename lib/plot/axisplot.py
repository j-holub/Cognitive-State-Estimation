import matplotlib.pyplot as plt
import numpy as np

def axisplot(data: list, labels: list):
    assert len(data) == len(labels)

    data_length = len(data[0])
    for d in data[1:]:
        assert len(d) == data_length

    # epochs on the x axis
    x_axis = np.arange(len(data[0]), dtype=np.uint8)+1

    # add every data entry with the respective label
    for i in range(len(data)):
        plt.plot(x_axis, data[i], label=labels[i])

    plt.legend()
