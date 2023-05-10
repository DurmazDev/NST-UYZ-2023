import glob
import argparse

parser = argparse.ArgumentParser(
    prog='list_classes.py',
    description='List class id\'s from labels folder.',
)

parser.add_argument('-d', '--dir', help='Directory of yolo labels', dest='dir', required=True)
args = parser.parse_args()

classes = []

for i in glob.glob(str(args.dir) + '/*.txt'):
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    for j in lines:
        j = j.replace('\n', '')
        j = j.split(' ')[0]
        try:
            classes.index(j)
        except ValueError:
            print(i)
            classes.append(j)

print(classes)
