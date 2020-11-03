import pandas as pd
import csv
import os
import sys

project_name = sys.argv[1]

file_name = './output/tps/' + project_name + '.csv'
tps = pd.read_csv(file_name).values

with open('./output/tps' + project_name + '_hunk.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    header = ['Project', 'BFC_SHA', 'BFC_Path', 'BFC_hunk']

    csv_writer.writerow(header)

    for i in range(len(tps)):
        project = tps[i][0]
        sha = tps[i][1]
        path = tps[i][2]

        stream = os.popen('cd /data/AllBIC/reference/repositories/' + project + ' ; '
                          'git checkout ' + sha + ' ; '
                          'git diff ' + sha + '~ ' + path)

        hunk = str(stream.read())

        line = [sha, path, hunk]

        csv_writer.writerow(line)

