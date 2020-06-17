import csv
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential

import logging
import numpy as np
import pandas as pd
import pickle
import platform
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def write_result_distance(out_file, distances, indices, testY):
    score = 0
    with open(out_file, 'w+') as file:
        file.write('real answers\n')
        for test in testY:
            file.write(str(test) + '\n')
        file.write('\n\npredicted answers\n')
        for i in range(len(indices)):
            # if distances[i] < 0.01:
            if np.any(distances[i] < 0.010):
                score += 1
            file.write(str(i) + ': instance:' + str(indices[i]) + ' distance: ' + str(distances[i]) + '\n\n')
        file.close()
    print('writing distances on', out_file, 'complete!')
    print('score:', score)
    return


def write_prediction(out_file, testX, testY, classifier):
    with open(out_file, 'w+') as file:
        file.write('real answers\n')
        for test in testY:
            file.write(str(test) + '\n')
        file.write('\n\npredicted answers\n')
        for yhat in classifier.predict(testX):
            file.write(yhat + '\n\n')
        file.close()
        print('writing predictions on', out_file, 'complete!')


def write_predict_proba(out_file, testX, classifier):
    proba = classifier.predict_proba(testX)
    vecs_on_csv(out_file, proba.T)
    print('writing test on', out_file, 'complete!')
    return


def write_kneighbors(out_file, testX, classifier):
    score = 0
    kneighbors = classifier.kneighbors(testX)
    with open(out_file, 'w') as fp:
        for i in range(len(kneighbors[0])):
            if np.any(kneighbors[0][i] < 0.001):
                score += 1
            fp.write(str(i) + ': ' + str(kneighbors[0][i]) + ' ' + str(kneighbors[1][i]) + '\n')
        # fp.write(str(kneighbors))
    print('score:', score)
    print('writing test on', out_file, 'complete!')
    return


def read_commits(input_file):
    """
    Parses each commits from corpus and returns info.

    Args:   input_file      (string):          file name of commit corpus

    Returns:
            commit_list     (list of string):  commits(list of string),
            commit_count    (int):             total number of counts,
            longest_len     (int):             length of longest commit
    """

    commit_corpus_file = open(input_file, 'r')
    commit_list = []
    is_start = False
    a_commit = ""
    commit_count = 0
    longest_len = 0

    for line in commit_corpus_file:
        if line.startswith('<SOC>'):
            is_start = True
        if is_start:
            a_commit += line
        if line.startswith('<EOC>'):
            commit_list.append(a_commit)
            if len(a_commit.split()) > longest_len:
                longest_len = len(a_commit.split())
            a_commit = ""
            commit_count += 1
            is_start = False

    return commit_list, commit_count, longest_len


if __name__ == '__main__':
    ##########################################################################
    # DATA PREPARATION

    # load commit String
    X_train, train_count, train_max = read_commits('inputs/string/S_train.txt')
    X_test, test_count, test_max = read_commits('inputs/string/S_calcite.txt')
    Y_train = pd.read_csv('inputs/string/YS_train.csv').values
    Y_train = pd.read_csv('inputs/string/YS_calcite.csv').values

    # pre-settings for LSTM model
    vocab_size = 5000
    encoded_train = [one_hot(d, vocab_size) for d in X_train]
    encoded_test = [one_hot(d, vocab_size) for d in X_test]

    print(np.array(X_train).shape)
    print(np.array(encoded_train).shape)
    print(np.array(X_test).shape)
    print(np.array(encoded_test).shape)

    # defining the model
    lstm = Sequential()
    lstm.add(Embedding(vocab_size, 512, input_length=None))
    lstm.add(LSTM(1024))
    lstm.add(Dense(1, activation='sigmoid'))
    lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(lstm.summary())
    # lstm.fit(encoded_train, Y_train, verbose=1)


    # Y_train_label = Y_train[:, 1]
    #
    # # training autoencoder
    # autoencoder = Model(input_commit, decoded)
    # autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')
    #
    # autoencoder.fit(X_train, X_train, epochs=15, batch_size=256, shuffle=True)
    #
    # T_autoencoder = autoencoder
    # T_encoder = Model(inputs=T_autoencoder.input, outputs=T_autoencoder.get_layer('encoder').output)
    #
    # # encoding dataset
    # X_train_encoded = T_encoder.predict(X_train)
    # X_test_encoded = T_encoder.predict(X_test)
    #
    # print('\nX_encoded:', X_train_encoded.shape)
    #
    # # training encoder + knn classifier
    # knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', algorithm='auto', weights='distance')
    # knn_n_encoder = knn.fit(X_train_encoded, Y_train_label)
    #
    # ##########################################################################
    # # Model Evaluation
    #
    # write_prediction('./eval/lstm_gumvec_prd.txt', X_test_encoded, Y_test, knn_n_encoder)
    # write_predict_proba('eval/lstm_gumvec_prob.txt', X_test_encoded, knn_n_encoder)
    # write_kneighbors('eval/lstm_gumvec_kneighbors.txt', X_test_encoded, knn_n_encoder)

    print('program finished!')
