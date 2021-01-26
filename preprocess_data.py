import csv
import numpy as np
import pandas as pd
import sys


# remove duplicate X and Y of dataframe
def rm_dups(X_file_path, Y_file_path):
    # f_trainX = open(X_file_path, 'r')
    # X_train = csv.reader(f_trainX)
    # X_train = np.asarray(list(X_train))
    # X_train = list(map(int, X_train))

    # Loop the data lines
    with open(X_file_path, 'r') as temp_f:
        # get No of columns in each line
        col_count = [len(l.split(",")) for l in temp_f.readlines()]

    # Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
    column_names = [i for i in range(0, max(col_count))]

    # Read csv
    X_train = pd.read_csv(X_file_path, header=None, delimiter=",", names=column_names)
    Y_train = pd.read_csv(Y_file_path, header=None)

    dup_idx = np.asarray(X_train.duplicated())  # must be before drop_duplicates
    X_train.drop_duplicates(inplace=True)

    i = 0
    for idx in range(len(dup_idx)):
        if dup_idx[idx]:
            Y_train.drop(Y_train.index[i], inplace=True)
            i -= 1
        i += 1

    # new_Y_train = []
    # X_train, dup_idx = np.unique(X_train, axis=0, return_index=True)
    # for idx in dup_idx:
    #     new_Y_train.append(Y_train[idx])

    return X_train, Y_train


def main(argv):
    file_path_X = './output/trainset/X_no_test_all.csv'
    file_path_Y = './output/trainset/Y_no_test_all.csv'

    trainX, trainY = rm_dups(file_path_X, file_path_Y)

    print('X_train.shape:', trainX.shape)
    print('Y_train.shape:', trainY.shape)


if __name__ == '__main__':
    main(sys.argv)
