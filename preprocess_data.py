import csv
import numpy as np
import pandas as pd
import sys


# remove duplicate X and Y of dataframe
def rm_dups(X_file_path, Y_file_path):
    X_train = pd.read_csv(X_file_path, header=None)
    Y_train = pd.read_csv(Y_file_path, header=None)

    dup_idx = np.asarray(X_train.duplicated())  # must be before drop_duplicates
    X_train.drop_duplicates(inplace=True)

    i = 0
    for idx in range(len(dup_idx)):
        if dup_idx[idx]:
            Y_train.drop(Y_train.index[i], inplace=True)
            i -= 1
        i += 1

    return X_train, Y_train


def main(argv):
    file_path_X = './output/trainset/X_no_test_all.csv'
    file_path_Y = './output/trainset/Y_no_test_all.csv'

    trainX, trainY = rm_dups(file_path_X, file_path_Y)

    print('X_train.shape:', trainX.shape)
    print('Y_train.shape:', trainY.shape)




if __name__ == '__main__':
    main(sys.argv)
