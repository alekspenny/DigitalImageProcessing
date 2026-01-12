# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import numpy as np
import cv2 as cv
# from google.colab.patches import cv2_imshow


def print_lab1(name):
    # Use a breakpoint in the code line below to debug your script.
    rnd = np.random.default_rng()
    # 1 array
    first = rnd.integers(0, 10, 10)
    # 2 array
    second = rnd.integers(0, 10, 10)

    print("#1 array", first)
    print("#2 array", second)

    result = np.array([first[i] for i in range(len(first)) if first[i] == second[i]])

    print(result)

    #
    elements_from_first_in_second = first[np.isin(first, second)]
    print("Elements from first that are in second:", elements_from_first_in_second)


def print_lab2(name):
    # download image
    image1 = cv.imread('/lenna.png')

    import matplotlib.pyplot as plt
    # %matplotlib inline позволяет выводить графики matplotlib в Jupyter
    plt.imshow(image1[:, :, :])

    histSize = [256]
    range = [0, 256]

    def plot_rgb_hist(image, histSize, range):
        histSize = [256]
        range = [0, 256]
        for i, col in enumerate(['b', 'g', 'r']):
            hist = cv.calcHist([image], [i], None, histSize, range)
            plt.plot(hist, color=col)
            plt.xlim(range)

    plot_rgb_hist(image1, histSize, range)
    plt.show()

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    plt.imshow(rgb_image1)

    rgb_result_image = np.empty(np.shape(rgb_image1), np.uint8)
    rgb_result_image[:, :, 0] = clahe.apply(rgb_image1[:, :, 0])
    rgb_result_image[:, :, 1] = clahe.apply(rgb_image1[:, :, 1])
    rgb_result_image[:, :, 2] = clahe.apply(rgb_image1[:, :, 2])

    gs = plt.GridSpec(2, 2)
    plt.figure(figsize=(10, 8))
    plt.subplot(gs[0])
    plt.imshow(rgb_image1)
    plt.subplot(gs[1])
    plt.imshow(rgb_result_image)
    plt.subplot(gs[2])
    plot_rgb_hist(rgb_image1, histSize, range)
    plt.subplot(gs[3])
    plot_rgb_hist(rgb_result_image, histSize, range)

    plt.show()
    print("end")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_lab1('PyCharm')
    print_lab2('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
