import csv
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import sys


# remove duplicate X and Y of dataframe
def rm_dups(X_train, Y_train):
    # dup_idx = np.asarray(X_train.duplicated())  # must be before drop_duplicates
    # X_train.drop_duplicates(inplace=True)
    #
    # i = 0
    # for idx in range(len(dup_idx)):
    #     if dup_idx[idx]:
    #         Y_train.drop(Y_train.index[i], inplace=True)
    #         i -= 1
    #     i += 1

    new_Y_train = []
    X_train, dup_idx = np.unique(X_train, axis=0, return_index=True)
    for idx in dup_idx:
        new_Y_train.append(Y_train[idx])

    return X_train, new_Y_train


def loadGumVec(train_file, train_label, test_file, test_label):
    f_trainX = open(train_file, 'r')
    trainX = csv.reader(f_trainX)
    f_testX = open(test_file, 'r')
    testX = csv.reader(f_testX)

    trainX = np.asarray(list(trainX))
    trainY = pd.read_csv(train_label, names=['index',
                                             'path_BBIC',
                                             'path_BIC',
                                             'sha_BBIC',
                                             'sha_BIC',
                                             'path_BBFC',
                                             'path_BFC',
                                             'sha_BBFC',
                                             'sha_BFC',
                                             'key',
                                             'project',
                                             'label'])
    testX = np.asarray(list(testX))
    testY = pd.read_csv(test_label, names=['index',
                                           'path_BBIC',
                                           'path_BIC',
                                           'sha_BBIC',
                                           'sha_BIC',
                                           'path_BBFC',
                                           'path_BFC',
                                           'sha_BBFC',
                                           'sha_BFC',
                                           'key',
                                           'project',
                                           'label'])
    train_max = 0
    test_max = 0
    # get the max length of vecs
    for i in range(len(trainX)):
        if train_max < len(trainX[i]):
            train_max = len(trainX[i])
    for i in range(len(testX)):
        if test_max < len(testX[i]):
            test_max = len(testX[i])

    # apply zero padding for fix vector length
    for i in range(len(trainX)):
        for j in range(len(trainX[i])):
            if trainX[i][j] == '':
                trainX[i][j] = 0
            else:
                trainX[i][j] = int(trainX[i][j])
        for j in range(train_max - len(trainX[i])):
            trainX[i].append(0)
    for i in range(len(testX)):
        for j in range(len(testX[i])):
            if testX[i][j] == '':
                testX[i][j] = 0
            else:
                testX[i][j] = int(testX[i][j])
        for j in range(test_max - len(testX[i])):
            testX[i].append(0)

    trainX = pad_sequences(trainX, padding='post')
    testX = pad_sequences(testX, padding='post')

    new_trainX = None
    new_testX = None

    # unifying vec length of train and test
    if train_max >= test_max:
        new_trainX = np.zeros(shape=(len(trainX), train_max))
        for i in range(len(trainX)):
            new_trainX[i] = np.asarray(trainX[i])
        new_testX = np.zeros(shape=(len(testX), train_max))
        for i in range(len(testX)):
            new_testX[i] = np.concatenate(
                [testX[i], np.zeros(shape=(train_max - test_max))])
    if test_max > train_max:
        new_trainX = np.zeros(shape=(len(trainX), test_max))
        new_testX = np.zeros(shape=(len(testX), test_max))
        for i in range(len(testX)):
            new_testX[i] = np.asarray(testX[i])
        for i in range(len(trainX)):
            new_trainX[i] = np.concatenate(
                [trainX[i], np.zeros(shape=(test_max - train_max))])

    f_trainX.close()
    f_testX.close()

    return new_trainX, trainY.values, new_testX, testY.values


def rm_buggy_from_clean(X_buggy, X_clean, Y_clean):
    # for i in range(len(X_buggy)):
    #     vec_buggy = np.asarray(X_buggy[i])
    #     for j in range(len(X_clean)):
    #         vec_clean = np.asarray(X_clean[j])
    #         # print((np.equal(vec_buggy, vec_clean)).all(), i, j)
    #         if np.equal(vec_buggy, vec_clean).all():
    #             X_clean = np.delete(X_clean, j, axis=0)
    #             Y_clean = np.delete(Y_clean, j, axis=0)
    #             break

    i = 0
    len_buggy = len(X_buggy)
    len_clean = len(X_clean)
    while i < len_buggy:
        vec_buggy = np.asarray(X_buggy[i])
        j = 0
        while j < len_clean:
            vec_clean = np.asarray(X_clean[j])
            # print((np.equal(vec_buggy, vec_clean)).all(), i, j)
            if np.equal(vec_buggy, vec_clean).all():
                X_clean = np.delete(X_clean, j, axis=0)
                Y_clean = np.delete(Y_clean, j, axis=0)
                len_clean -= 1
                j -= 1
                break
            j += 1
        i += 1

    return X_clean, Y_clean


def main(argv):
    # load X Y data for both buggy and clean
    buggyX, buggyY, cleanX, cleanY = loadGumVec(
        './output/trainset/test/X_no_test_buggy.csv',
        './output/trainset/test/Y_no_test_buggy.csv',
        './output/trainset/test/X_no_test_clean.csv',
        './output/trainset/test/Y_no_test_clean.csv'
    )

    # remove duplicated vectors from clean data set
    cleanX,  cleanY = rm_dups(cleanX, cleanY)
    buggyY = np.asarray(buggyY)
    cleanY = np.asarray(cleanY)

    print(type(buggyX[0]))

    print()
    print('<< After rm_dups >>')
    print('buggyX.shape:', buggyX.shape)
    print('buggyY.shape:', buggyY.shape)
    print('cleanX.shape:', cleanX.shape)
    print('cleanY.shape:', cleanY.shape)
    print()

    # remove the instances that are the same as buggy in clean data set
    cleanX, cleanY = rm_buggy_from_clean(buggyX, cleanX, cleanY)

    print('<< After rm_buggy_from_clean >>')
    print('buggyX.shape:', buggyX.shape)
    print('buggyY.shape:', buggyY.shape)
    print('cleanX.shape:', cleanX.shape)
    print('cleanY.shape:', cleanY.shape)


if __name__ == '__main__':
    main(sys.argv)
