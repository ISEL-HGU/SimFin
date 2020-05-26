import csv
from gensim.models import Word2Vec
from keras.layers import Input, Dense
from keras.models import Model
import logging
import numpy as np
import pandas as pd
import pickle
import platform
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# dividing scenario of server and mac
W2V_DIM = 0
if platform.system() == 'Linux':
    W2V_DIM = 1
else:
    W2V_DIM = 1

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

# Global variable: the size of word2vector


class W2VModel:
    def __init__(self, commits, file_path):
        self.commits = commits
        self.file_path = file_path
        self.model = None

    def saveW2V(self):
        model = Word2Vec(self.commits,
                         size=W2V_DIM,
                         window=10,
                         min_count=1,
                         workers=32,
                         sg=1)
        model.save(self.file_path)
        self.model = model
        print('saving', self.file_path, 'complete!')
        return

    def loadW2V(self):
        self.model = Word2Vec.load(self.file_path)
        print('loading', self.file_path, 'complete!')
        return

    def trainFromLoad(self, newList):
        self.model.train(newList, total_examples=10, epochs=10, word_count=0)
        return


class CommitVectors:
    def __init__(self, commits, commit_count, longest_len):
        self.commits = commits
        self.commit_count = commit_count
        self.longest_len = longest_len
        self.vectorized_commits = 0

    def vectorize_commits(self, wv):
        """
        Vectorizes the commit corpus w.r.t. w2v model

        Args:   wv  (w2v model):    w2v trained from corpus

        Returns vectorized_commits ([commit_count][longest_len][1])
        """
        commits = np.asarray(self.commits)
        vectorized_commits = []
        for commit in commits:
            vector_of_commit = []
            for token in commit:
                vector_of_token = wv[token]
                for val in vector_of_token:
                    vector_of_commit.append(val)
            if len(commit) < self.longest_len:
                for i in range(self.longest_len - len(commit)):
                    for j in range(W2V_DIM):
                        vector_of_commit.append(0.0)
            vectorized_commits.append(vector_of_commit)

        self.vectorized_commits = vectorized_commits
        print('vectorizing commits complete!')
        return

    def write_cv(self, filePath):
        file = open(filePath, 'wb')
        pickle.dump(self.vectorized_commits, file)
        file.close()
        print('writing commit vectors complete!')
        return

    def read_cv(self, filePath):
        file = open(filePath, 'rb')
        cvs = pickle.load(file)
        file.close()
        self.vectorized_commits = cvs
        print('reading commit vectors complete!')
        return cvs


def write_result_distance(out_file, distances, indices, testY):
    score = 0
    with open(out_file, 'w+') as file:
        file.write('real answers\n')
        for test in testY:
            file.write(str(test) + '\n')
        file.write('\n\npredicted answers\n')
        for i in range(len(indices)):
            if distances[i] < 0.01:
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
            file.write(yhat+'\n\n')
        file.close()
        print('writing predictions on', out_file, 'complete!')


def write_predict_proba(out_file, testX, classifier):
    proba = classifier.predict_proba(testX)
    vecs_on_csv(out_file, proba.T)
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
        if line.startswith('<start>'):
            is_start = True
        if is_start:
            a_commit += line
        if line.startswith('<end>'):
            commit_list.append(a_commit.split())
            if len(a_commit.split()) > longest_len:
                longest_len = len(a_commit.split())
            a_commit = ""
            commit_count += 1
            is_start = False

    return commit_list, commit_count, longest_len


def vecs_on_csv(filePath, X_dbn):
    # writing out the features learned by dbn on a csv file
    df = pd.DataFrame(data=X_dbn[0:][0:],
                      index=[i for i in range(X_dbn.shape[0])],
                      columns=['f' + str(i) for i in range(X_dbn.shape[1])])
    df.to_csv(filePath)
    return


def loadW2V(train_file, test_file, combined, train_label, test_label):
    # read BIC corpus from txt file

    # parses the commit corpus
    train_commits, train_com_cnt, train_lgst_len = read_commits(train_file)
    test_commits, test_com_cnt, test_lgst_len = read_commits(test_file)
    combined_commits, combined_com_cnt, combined_lgst_len = read_commits(
        combined)

    # train/load W2VModel
    w2v = W2VModel(combined_commits, './word2vec/combined_w2v.model')
    w2v.saveW2V()
    # w2v.loadW2V()

    # vectorizing the corpus with w2v model
    train_cvs = CommitVectors(train_commits, train_com_cnt, combined_lgst_len)
    train_cvs.vectorize_commits(w2v.model.wv)
    train_cvs.write_cv('./commit_vec/train_cvs.pkl')
    # train_cvs.read_cv('./commit_vec/train_cvs.pkl')

    test_cvs = CommitVectors(test_commits, test_com_cnt, combined_lgst_len)
    test_cvs.vectorize_commits(w2v.model.wv)
    test_cvs.write_cv('./commit_vec/test_cvs.pkl')
    # test_cvs.read_cv('./commit_vec/test_cvs.pkl')

    # data before
    train_cv = train_cvs.vectorized_commits
    test_cv = test_cvs.vectorized_commits

    train_cv = np.asarray(train_cv)
    train_label = pd.read_csv(train_label)
    test_cv = np.asarray(test_cv)
    test_label = pd.read_csv(test_label)

    return train_cv, train_label, test_cv, test_label


def loadGumVec(train_file, train_label, test_file, test_label):
    f_trainX = open(train_file, 'r')
    trainX = csv.reader(f_trainX)
    f_testX = open(test_file, 'r')
    testX = csv.reader(f_testX)

    trainX = np.asarray(list(trainX))
    trainY = pd.read_csv(train_label)
    testX = np.asarray(list(testX))
    testY = pd.read_csv(test_label)

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
        for j in range(train_max - len(trainX[i])):
            trainX[i].append(0)
    for i in range(len(testX)):
        for j in range(test_max - len(testX[i])):
            testX[i].append(0)

    # if the vec is '' change to 0
    for i in range(len(trainX)):
        for j in range(len(trainX[i])):
            if trainX[i][j] == '':
                trainX[i][j] = 0
            else:
                trainX[i][j] = int(trainX[i][j])
    for i in range(len(testX)):
        for j in range(len(testX[i])):
            if testX[i][j] == '':
                testX[i][j] = 0
            else:
                testX[i][j] = int(testX[i][j])

    dim_train = len(trainX[0])
    dim_test = len(testX[0])
    new_trainX = None
    new_testX = None

    # fixing vec length for train and test set
    if dim_train >= dim_test:
        new_trainX = trainX
        new_testX = np.zeros(shape=(len(testX), len(trainX[0])))
        for i in range(len(testX)):
            new_testX[i] = np.concatenate([testX[i], np.zeros(shape=(dim_train - dim_test))])

    if dim_test > dim_train:
        new_trainX = np.zeros(shape=(len(trainX), len(testX[0])))
        new_testX = testX
        for i in range(len(trainX)):
            new_trainX[i] = np.concatenate([trainX[i], np.zeros(shape=(dim_test - dim_train))])

    f_trainX.close()
    f_testX.close()

    print(train_max)
    print(test_max)

    return new_trainX, trainY.values, new_testX, testY.values


if __name__ == '__main__':
    ##########################################################################
    # DATA PREPARATION

    # # load vectors using word2vec
    # X_train, Y_train, X_test, Y_test = loadW2V(
    #     './inputs/old/100_code.txt',
    #     './inputs/old/math_code.txt',
    #     './inputs/old/100_math_code.txt',
    #     './inputs/old/100_label.csv',
    #     './inputs/old/math_label.csv')

    # load Gumtree Vectors
    X_train, Y_train, X_test, Y_test = loadGumVec(
        './inputs/DataCollector/GVNC_train.csv',
        './inputs/DataCollector/Y_train.csv',
        './inputs/DataCollector/GVNC_zookeeper.csv',
        './inputs/DataCollector/Y_zookeeper.csv'
    )

    print(X_train.shape)
    print(X_train[0].shape)

    # splitting test from one set
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train,
    #                                                     test_size=0.005,
    #                                                     random_state=0)
    Y_train_label = Y_train[:, 1]

    ##########################################################################
    # Model Preparation

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    with open('./view_file/X_train.csv', '+w') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(X_train)

        wr.writerows(X_train)

    with open('./view_file/X_test.csv', '+w') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(X_test)

    # print('feature size: ', X_train.shape[1])
    print('\noriginal train data X (vectorized): ', X_train.shape)
    print('\noriginal test data X (vectorized): ', X_test.shape)

    # Preparing Deep AE
    feature_dim = X_train.shape[1]
    input_commit = Input(shape=(feature_dim,))
    encoded = Dense(500, activation='relu')(input_commit)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(500, activation='relu', name='encoder')(encoded)

    decoded = Dense(500, activation='relu')(encoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(feature_dim, activation='sigmoid')(decoded)

    ##########################################################################
    # Model Training

    # training autoencoder
    autoencoder = Model(input_commit, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta')

    autoencoder.fit(X_train, X_train, epochs=15, batch_size=256, shuffle=True)

    T_autoencoder = autoencoder
    T_encoder = Model(inputs=T_autoencoder.input, outputs=T_autoencoder.get_layer('encoder').output)

    # encoding dataset
    X_train_encoded = T_encoder.predict(X_train)
    X_test_encoded = T_encoder.predict(X_test)

    # wrting encoded dataset for checking
    vecs_on_csv('./view_file/train_encoded.csv', X_train_encoded)
    vecs_on_csv('./view_file/test_encoded.csv', X_test_encoded)

    print('\nX_encoded:', X_train_encoded.shape)

    # training encoder + knn classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan', algorithm='auto')
    knn_n_encoder = knn.fit(X_train_encoded, Y_train_label)

    # training Nearest Neighbours for distance
    # nbrs_vanilla = NearestNeighbors(n_neighbors=1, metric='manhattan', algorithm='auto').fit(X_train)
    # distances_vnla, indices_vnla = nbrs_vanilla.kneighbors(X_test)
    nbrs_encoder = NearestNeighbors(n_neighbors=1, metric='manhattan', algorithm='auto').fit(X_train_encoded)
    distances_encd, indices_encd = nbrs_encoder.kneighbors(X_test_encoded)

    ##########################################################################
    # Model Evaluation

    write_result_distance('./eval/encoder_gumvec_dst.txt', distances_encd, indices_encd, Y_test)

    write_prediction('./eval/encoder_gumvec_prd.txt', X_test_encoded, Y_test, knn_n_encoder)

    print('program finished!')
