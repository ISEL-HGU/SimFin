import csv
import os
import pandas as pd
import sys


# python3 ./SimFin/sort_top_k.py preprocessed ranger
def main(argv):
    train_name = argv[1]
    test_name = argv[2]
    file_path = '/data/jihoshin/' + train_name + '/' + test_name
    Y_train = pd.read_csv('./output/trainset/Y_' + train_name + '.csv',
                          names=['index', 'path_BBIC', 'path_BIC', 'sha_BBIC', 'sha_BIC',
                                 'path_BBFC', 'path_BFC', 'sha_BBFC', 'sha_BFC', 'key',
                                 'project', 'label']).values
    print('Y_train:', len(Y_train))
    test_num = len([name for name in os.listdir(file_path)])
    print('test_num:', test_num)
    for i in range(test_num):
        test_path = file_path + '/test' + str(i)
        dist_i = pd.read_csv(test_path + '/distance.csv', header=None).values
        test_dict = dict()
        # print('dist_len:', len(dist_i))
        for j in range(len(dist_i)):
            test_dict[str(j)] = dist_i[j]
        sorted_test = sorted(test_dict.items(), key=lambda item: item[1])
        with open(test_path + '/sorted.csv', 'w', newline='') as sorted_csv:
            csv_writer = csv.writer(sorted_csv, delimiter=',')
            # print(len(sorted_test))
            for (key, value) in sorted_test:
                key = int(key)
                value = float(value)
                yhat_label = Y_train[key][11]
                row = [value, key, yhat_label]
                csv_writer.writerow(row)
        print('test', i, 'done!')


if __name__ == '__main__':
    main(sys.argv)
