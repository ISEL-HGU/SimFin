import os
import csv
import getopt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
from sklearn.preprocessing import MinMaxScaler
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K_NEIGHBORS = 1

np.set_printoptions(threshold=np.inf)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


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
                                             'sha_BBFC'
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
                                           'sha_BBFC'
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


def write_result(file, results):

    with open(file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        header = []
        for i in range(len(results)):
            header.append('test' + i)
        csv_writer.writerow(header)

        csv_writer.writerows(results)

    return


def calculate_manhattan(trainX, testX):
    # manhattan = [[0 for c in range(len(testX))] for r in range(len(trainX))]
    # for i in range(len(testX)):
    #     for j in range(len(trainX)):
    #         manhattan[j][i] = distance.cityblock(testX[i], trainX[j])
    #     print(i, "/", len(testX), sep='')
    manhattan = distance.cdist(testX, trainX, 'cityblock')

    return manhattan


def main(argv):
    global K_NEIGHBORS
    train_name = 'no_input_for_train'
    test_name = 'no_input_for_test'
    seed = 3

    try:
        opts, args = getopt.getopt(argv[1:], "ht:k:p:s:", ["help", "train", "k_neighbors", "predict", "seed"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print("")
            sys.exit()
        elif o in ("-t", "--train"):
            train_name = a
        elif o in ("-k", "--k_neighbors"):
            K_NEIGHBORS = int(a)
        elif o in ("-p", "--predict"):
            test_name = a
        elif o in ("-s", "--seed"):
            seed = a
        else:
            assert False, "unhandled option"

    # 1. load vectors
    trainX, trainY, testX, testY = loadGumVec(
        './output/trainset/X_' + train_name + '.csv',
        './output/trainset/Y_' + train_name + '.csv',
        './output/testset/X_' + test_name + '.csv',
        './output/testset/Y_' + test_name + '.csv'
    )

    ##########################################################################
    # DATA PREPARATION

    print('original X_train.shape: ', trainX.shape)
    print('original Y_train.shape: ', trainY.shape)

    Y_train_label = trainY[:, 8]

    ##########################################################################
    # Model Preparation

    # 2. apply scaler to both train & test set
    scaler = MinMaxScaler()
    scaler.fit(trainX)

    X_train = scaler.transform(trainX)
    X_test = scaler.transform(testX)

    ##########################################################################
    # Model Evaluation

    # 3. load AED model
    encoder = load_model('./PatchSuggestion/models/' + train_name + str(seed) + '_encoder.model', compile=False)

    # 4. encode train & test set
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # 5. distance calculation
    # distance_result = calculate_manhattan(X_train_encoded, X_test_encoded)
    distance_result = distance.cdist(X_train_encoded, X_test_encoded, 'cityblock')

    # writing the result of knn prediction
    resultFile = './output/eval/' + test_name + '_' + train_name + '_' + str(seed) + '_result.csv'
    write_result(resultFile,
                 distance_result)

    print('loaded and predicted ' + test_name + '_' + train_name + '_' + str(seed) + '_result.csv complete!')


if __name__ == '__main__':
    main(sys.argv)
