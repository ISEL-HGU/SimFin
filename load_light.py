import os
import csv
import getopt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np
import pandas as pd
import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# np.set_printoptions(threshold=np.inf)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


# writing out the features learned by the model on a csv file
def vecs_on_csv(filePath, X_dbn):
    df = pd.DataFrame(data=X_dbn[0:][0:])
    df.to_csv(filePath, index=False, header=False)
    return


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def del_index_num(s):
    temp = ''
    is_passed = False
    for c in reversed(s):
        if is_number(c) and not is_passed:
            continue
        else:
            temp += c
            is_passed = True

    temp = list(temp)
    temp.reverse()
    s = ''.join(temp)

    return s


def write_kneighbors(out_file, testX, classifier):
    score = 0
    kneighbors = classifier.kneighbors(testX)
    with open(out_file, 'w') as fp:
        for i in range(len(kneighbors[0])):
            if np.any(kneighbors[0][i] < 0.001):
                score += 1
            fp.write(str(i) + ': ' +
                     str(kneighbors[0][i]) + ' ' + str(kneighbors[1][i]) + '\n')
        # fp.write(str(kneighbors))
    print('score:', score)
    print('writing test on', out_file, 'complete!')
    return


def write_test_result(out_file, testX, classifier):
    with open(out_file, 'w+') as file:
        for yhat in classifier.predict(testX):
            file.write(yhat + '\n\n')
    print('writing test on', out_file, 'complete!')
    return


# def write_result(trainY, testY, out_file, testX, classifier):
#     kneibors = classifier.kneighbors(testX)

#     with open(out_file, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile, delimiter=',')

#         # writing header
#         header = ['Y_BIC_SHA', 'Y_BIC_Path', 'Y_BIC_Hunk',
#                   'Y_BFC_SHA', 'Y_BFC_Path', 'Y_BFC_Hunk',
#                   'Rank', 'Sim-Score', 'Label', 'Project',
#                   'Y^_BIC_SHA', 'Y^_BIC_Path', 'Y^_BIC_Hunk',
#                   'Y^_BFC_SHA', 'Y^_BFC_Path', 'Y^_BFC_Hunk']

#         csv_writer.writerow(header)

#         # writing each row values (test * (predicted * k_keighbors))
#         for i in range(len(testY)):
#             # writing real answer (y)
#             y_bic_sha = str(testY[i][3])
#             y_bic_path = str(testY[i][1])
#             y_bfc_sha = str(testY[i][7])
#             y_bfc_path = str(testY[i][5])
#             y_real_label = testY[i][10]

#             y_bic_hunk = '-'
#             y_bfc_hunk = '-'

#             # writing predicted answers (y^)
#             for j in range(K_NEIGHBORS):
#                 pred_idx = kneibors[1][i][j]
#                 yhat_project = trainY[pred_idx][9]
#                 yhat_bic_sha = str(trainY[pred_idx][3])
#                 yhat_bic_path = str(trainY[pred_idx][1])
#                 yhat_bfc_sha = str(trainY[pred_idx][7])
#                 yhat_bfc_path = str(trainY[pred_idx][5])

#                 yhat_bic_hunk = '-'
#                 yhat_bfc_hunk = '-'

#                 instance = [y_bic_sha, y_bic_path, y_bic_hunk,
#                             y_bfc_sha, y_bfc_path, y_bfc_hunk,
#                             j + 1, kneibors[0][i][j], y_real_label, yhat_project,
#                             yhat_bic_sha, yhat_bic_path, yhat_bic_hunk,
#                             yhat_bfc_sha, yhat_bfc_path, yhat_bfc_hunk]

#                 csv_writer.writerow(instance)



def load_gumvecs(test_file):
    f_testX = open(test_file, 'r')
    testX = csv.reader(f_testX)

    testX = np.asarray(list(testX))

    test_max = 0
    train_max = 551

    # get the max length of vecs
    for i in range(len(testX)):
        if test_max < len(testX[i]):
            test_max = len(testX[i])

    # apply zero padding for fix vector length
    for i in range(len(testX)):
        for j in range(len(testX[i])):
            if testX[i][j] == '':
                testX[i][j] = 0
            else:
                testX[i][j] = int(testX[i][j])
        for j in range(test_max - len(testX[i])):
            testX[i].append(0)

    testX = pad_sequences(testX, padding='post')

    new_testX = None

    # unifying vec length of train and test
    if train_max >= test_max:
        new_testX = np.zeros(shape=(len(testX), train_max))
        for i in range(len(testX)):
            new_testX[i] = np.concatenate(
                [testX[i], np.zeros(shape=(train_max - test_max))])
    if test_max > train_max:
        new_testX = np.zeros(shape=(len(testX), test_max))
        for i in range(len(testX)):
            new_testX[i] = np.asarray(testX[i])

    f_testX.close()
    return new_testX


def write_pickle(src, filePath):
    file = open(filePath, 'wb')
    pickle.dump(src, file, protocol=4)
    file.close()
    print('writing on', filePath, 'complete!')
    return


def load_pickle(filePath):
    file = open(filePath, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def main(argv):
    train_name = 'no_input_for_train'
    test_name = 'no_input_for_test'
    version_name = 'no_version_name'

    try:
        opts, args = getopt.getopt(argv[1:], "ht:p:v:", ["help", "train", "predict", "version"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print("")
            sys.exit()
        elif o in ("-t", "--train"):
            train_name = a
        elif o in ("-p", "--predict"):
            test_name = a
        elif o in ("-v", "--version"):
            version_name = a
        else:
            assert False, "unhandled option"



    ##########################################################################
    # DATA PREPARATION
    # 1. load vectors
    testX = load_gumvecs(
        './output/testset/X_' + test_name + '.csv',
    )

    ##########################################################################
    # Model Preparation

    # 2. apply scaler to both train & test set

    scaler = load_pickle('./output/models/' + train_name + '_scaler.pkl')
    X_test = scaler.transform(testX)

    ##########################################################################
    # Model Evaluation

    # 3. load AED model
    encoder = load_model('./output/models/' + train_name + '3_encoder.model', compile=False)

    # 4. encode test set and write them in view_file
    X_test_encoded = encoder.predict(X_test)
    vecs_on_csv('./output/view_file/' + test_name + '_' + version_name +'_encoded.csv', X_test_encoded)

    # 5. apply kNN model
    # knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS,
    #                            metric='manhattan',
    #                            algorithm='kd_tree',
    #                            weights='distance')

    # knn.fit(X_train_encoded.astype(str), Y_train_label)

    # writing the result of knn prediction
    # resultFile = './output/eval/' + test_name + '_' + train_name + '_' + str(seed) + '_result.csv'
    # write_result(trainY,
    #              testY,
    #              resultFile,
    #              X_test_encoded,
    #              knn)

    print('loaded and predicted ' + test_name + '_' + train_name + '_result.csv complete!')


if __name__ == '__main__':
    main(sys.argv)
