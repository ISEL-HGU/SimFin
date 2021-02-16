import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


def main(argv):
    test_name = argv[1]
    test_index = argv[2]
    test_label = argv[3]
    file_path = '/data/jihoshin/' + test_name
    test_num = len([name for name in os.listdir(file_path)])

    # for i in range(test_num):
    test_path = file_path + '/test' + test_index
    sort_i_1000 = pd.read_csv(test_path + '/sorted.csv',
                              names=['distance', 'index', 'label'])

    len_list = []
    for i in range(100):
        len_list.append(i)

    buggy_df = sort_i_1000[sort_i_1000['label'] == 1]
    clean_df = sort_i_1000[sort_i_1000['label'] == 0]

    quo_b, mod_b = divmod(len(buggy_df), 100)
    quo_c, mod_c = divmod(len(clean_df), 100)

    print(quo_b, mod_b)
    print(quo_c, mod_c)

    bug_100 = []
    i = 1
    while i <= 100:
        j = 1
        avg_temp = 0
        while j <= quo_b:
            avg_temp += buggy_df.iloc[i * quo_b]['distance']
            j += 1
        bug_100.append(avg_temp / (j - 1))
        i += 1

    clean_100 = []
    i = 1
    while i <= 100:
        j = 1
        avg_temp = 0
        while j <= quo_c:
            avg_temp += clean_df.iloc[i * quo_c]['distance']
            j += 1
        clean_100.append(avg_temp / (j - 1))
        i += 1

    plt.plot(len_list, bug_100, label='buggy')
    plt.plot(len_list, clean_100, label='clean')
    plt.xlabel('sorted_index')
    plt.ylabel('distance')

    plt.legend()
    plt.show()
    plt.savefig('./output/inspection/' + test_name + '_' + test_label + '_' + test_index + '.png')


if __name__ == '__main__':
    main(sys.argv)
