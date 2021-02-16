import csv
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


def main(argv):
    test_name = argv[1]
    test_index = argv[2]
    file_path = '/data/jihoshin/' + test_name
    test_num = len([name for name in os.listdir(file_path)])

    # for i in range(test_num):
    test_path = file_path + '/test' + test_index
    sort_i_1000 = pd.read_csv(test_path + '/sorted.csv', names=['distance', 'index', 'label'], nrows=1000)

    fig, ax = plt.subplots()

    for key, group in sort_i_1000.groupby('label'):
        group.plot('distance', yerr='std', label=key, ax=ax)

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
