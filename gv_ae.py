import csv
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf)


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


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


def vecs_on_csv(filePath, X_dbn):
    # writing out the features learned by dbn on a csv file
    df = pd.DataFrame(data=X_dbn[0:][0:],
                      index=[i for i in range(X_dbn.shape[0])],
                      columns=['f' + str(i) for i in range(X_dbn.shape[1])])
    df.to_csv(filePath)
    return


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
            new_testX[i] = np.concatenate([testX[i], np.zeros(shape=(train_max - test_max))])
    if test_max > train_max:
        new_trainX = np.zeros(shape=(len(trainX), test_max))
        new_testX = np.zeros(shape=(len(testX), test_max))
        for i in range(len(testX)):
            new_testX[i] = np.asarray(testX[i])
        for i in range(len(trainX)):
            new_trainX[i] = np.concatenate([trainX[i], np.zeros(shape=(test_max - train_max))])

    f_trainX.close()
    f_testX.close()

    return new_trainX, trainY.values, new_testX, testY.values


def write_pickle(src, filePath):
    file = open(filePath, 'wb')
    pickle.dump(src, file)
    file.close()
    print('writing on', filePath, 'complete!')
    return


if __name__ == '__main__':
    ##########################################################################
    # DATA PREPARATION

    # load Gumtree Vectors
    X_train, Y_train, X_test, Y_test = loadGumVec(
        './inputs/apache/GVNC_train.csv',
        './inputs/apache/Y_train.csv',
        './inputs/apache/GVNC_calcite.csv',
        './inputs/apache/Y_calcite.csv'
    )

    print(X_train.shape)
    print(Y_train.shape)
    
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

    write_pickle(autoencoder, './models/ae.model')
    write_pickle(T_encoder, './models/encoder.model')

    # wrting encoded dataset for checking
    vecs_on_csv('./view_file/train_encoded.csv', X_train_encoded)
    vecs_on_csv('./view_file/test_encoded.csv', X_test_encoded)

    print('\nX_encoded:', X_train_encoded.shape)

    # training encoder + knn classifier
    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', algorithm='auto', weights='distance')
    knn_n_encoder = knn.fit(X_train_encoded, Y_train_label)

    write_pickle(knn_n_encoder, './models/knn.model')

    ##########################################################################
    # Model Evaluation

    write_kneighbors('eval/encoder_gumvec_kneighbors.txt', X_test_encoded, knn_n_encoder)

    print('program finished!')