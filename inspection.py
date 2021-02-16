import csv
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


def main(argv):
    test_name = argv[1]
    test_index = argv[2]
    file_path = './data/jihoshin/' + test_name
    test_num = len([name for name in os.listdir(file_path)])

    # for i in range(test_num):
    test_path = file_path + '/test' + test_index
    sort_i_1000 = pd.read_csv(test_path + '/sorted.csv',
                              names=['distance', 'index', 'label'])

    buggy_df = sort_i_1000[sort_i_1000['label'] == 1]
    clean_df = sort_i_1000[sort_i_1000['label'] == 0]

    bug_len_list = []
    clean_len_list = []
    for i in range(len(buggy_df)):
        bug_len_list.append(i)
    for i in range(len(clean_df)):
        clean_len_list.append(i)

    print('bug_len:', len(bug_len_list))
    print('clean_len:', len(clean_len_list))

    plt.plot(bug_len_list, buggy_df['distance'].tolist(), label='buggy')
    plt.plot(clean_len_list, clean_df['distance'].tolist(), label='clean')
    plt.xlabel('sorted_index')
    plt.ylabel('distance')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main(sys.argv)
