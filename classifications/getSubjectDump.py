import re
import json
import sys

if len(sys.argv) > 1:
    fpath = sys.argv[1]
else:
    fpath = 'galaxy-builder-subjects.csv'

try:
    with open(fpath) as f:
        classificationCsv = f.read().split('\n')[1:]
except FileNotFoundError:
    print('No subjects file found, exiting')
    sys.exit(0)
