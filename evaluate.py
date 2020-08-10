import numpy as np
import getopt
import pandas as pd
import sys

CUTOFF = 0
K_NEIGHBORS = 1


def evaluate(trainY, result_file):
    Y_train = pd.read_csv(trainY, names=['index',
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

    results = pd.read_csv(result_file, names=['Y_BIC_SHA', 'Y_BIC_Path', 'Y_BIC_Hunk',
                                              'Y_BFC_SHA', 'Y_BFC_Path', 'Y_BFC_Hunk',
                                              'Rank', 'Sim-Score', 'BI_lines', 'Label',
                                              'Y^_BIC_SHA', 'Y^_BIC_Path', 'Y^_BIC_Hunk',
                                              'Y^_BFC_SHA', 'Y^_BFC_Path', 'Y^_BFC_Hunk']).values
    # parse cutoffs
    cutoffs = str(CUTOFF).split(',')
    min_cutoff = float(cutoffs[0])
    max_cutoff = float(cutoffs[1])

    # instantiate list with dimension (cutoff, instance, k_neighbor)
    cutoff_idx = 0
    cutoff = min_cutoff
    while cutoff < max_cutoff:
        cutoff += 0.1
        cutoff_idx += 1
    distance_list = np.zeros((cutoff_idx, K_NEIGHBORS, len(results) / K_NEIGHBORS))
    prediction_list = np.zeros((cutoff_idx, K_NEIGHBORS, len(results) / K_NEIGHBORS))

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    cutoff_idx = 0
    cutoff = min_cutoff
    while cutoff < max_cutoff:
        try:
            for i in range(1, K_NEIGHBORS + 1):
                for j in range(len(results)):
                    if results[j][6] <= i:
                        distance_list[cutoff_idx][i][j] += (results[j][7] / j)
                    else:
                        continue
        finally:
            cutoff += 0.1
            cutoff_idx += 1

    cutoff_idx = 0
    cutoff = min_cutoff
    while cutoff < max_cutoff:
        try:
            for i in range(1, K_NEIGHBORS + 1):
                for j in range(len(results)):
                    if distance_list[cutoff_idx][i][j] < cutoff:
                        prediction_list[cutoff_idx][i][j] = 1
                    else:
                        prediction_list[cutoff_idx][i][j] = 0
        finally:
            cutoff += 0.1
            cutoff_idx += 1

    print('len of distance_list', len(distance_list))

    for i in range(len(Y_train)):
        label = int(Y_train[i][10])
        if prediction_list == 1 and label == 1:
            tp += 1
        elif prediction_list == 1 and label == 0:
            fp += 1
        elif prediction_list == 0 and label == 1:
            fn += 1
        elif prediction_list == 0 and label == 0:
            tn += 1

    return prediction_list


def write_recall(pred_list):
    cutoffs = str(CUTOFF).split(',')
    min_cutoff = float(cutoffs[0])
    max_cutoff = float(cutoffs[1])

    cutoff_idx = 0
    cutoff = min_cutoff
    while cutoff < max_cutoff:
        try:
            for i in range(1, K_NEIGHBORS + 1):
                for j in range(len(results)):
                    if results[j][6] <= i:
                        distance_list[cutoff_idx][i][j] += (results[j][7] / j)
                    else:
                        continue
        finally:
            cutoff += 0.1
            cutoff_idx += 1
    return


def write_precision(pred_list):
    return


def write_f_measure(pred_list):
    return


def main(argv):
    global CUTOFF
    global K_NEIGHBORS
    test_name = ''
    try:
        opts, args = getopt.getopt(argv[1:], "hk:c:t:", ["help", "k_neighbors", "cutoff", "test"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print("")
            sys.exit()
        elif o in ("-k", "--k_neighbors"):
            K_NEIGHBORS = int(a)
        elif o in ("-c", "--cutoff"):
            CUTOFF = a
        elif o in ("-t", "--test"):
            test_name = a
        else:
            assert False, "unhandled option"

    label_file = './output/testset/Y_' + test_name + '.csv'
    result_file = './output/eval/' + test_name + '_result.csv'

    evaluate(label_file, result_file)


if __name__ == '__main__':
    main(sys.argv)
