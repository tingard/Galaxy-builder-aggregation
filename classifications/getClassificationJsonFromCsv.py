import re
import json
import sys

if len(sys.argv) > 1:
    fpath = sys.argv[1]
else:
    fpath = 'galaxy-builder-classifications.csv'

try:
    with open(fpath) as f:
        classificationCsv = f.read().split('\n')[1:]
except FileNotFoundError:
    print('No classification file found, exiting')
    sys.exit(0)

if len(classificationCsv[-1]) == 0:
    classificationCsv.pop(-1)

classificationList = []
subjIdList = []

for cls in classificationCsv:
    c = re.search(',"\[\{""task.*\}\]\}\]', cls)
    subjIdList.append(int(cls.split(',')[-1]))
    if c:
        classificationList.append(
            json.loads(
                c.group()[2:]
                 .replace('""', '"')
            )
        )

print('Writing out {} classifications'.format(len(classificationList)))
with open('galaxy-builder-classifications.json', 'w') as f:
    json.dump(classificationList, f)

print('With {} corresponding subject IDs'.format(len(subjIdList)))
with open('subjectForClassification.csv', 'w') as f:
    f.write('\n'.join((str(i) for i in subjIdList)))


uniqueSubjectIDs = list(set(subjIdList))
print('Out of a total {} subjects'.format(len(uniqueSubjectIDs)))
with open('allSubjectIds.json', 'w') as f:
    json.dump(uniqueSubjectIDs, f)
