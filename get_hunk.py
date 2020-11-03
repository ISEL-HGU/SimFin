import pandas as pd
import csv
import os
import sys

project_name = sys.argv[1]

file_name = '/Users/jihoshin/Desktop/random3/tps/' + project_name + '.csv'
tps = pd.read_csv(file_name).values

with open('out', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    header = ['Y^_BFC_SHA', 'Y^_BFC_Path', 'Y^_BFC_hunk']

    csv_writer.writerow(header)

    for i in range(len(tps)):
        sha = tps[i][0]
        path = tps[i][1]

        stream = os.popen('cd /data/AllBIC/reference/repositories/' + project_name + ' ; '
                          'git checkout ' + sha + ' ; '
                          'git diff ' + sha + '~ ' + path)

        hunk = str(stream.read())

        line = [sha, path, hunk]

        csv_writer.writerow(line)