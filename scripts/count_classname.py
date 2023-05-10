import glob
import argparse

parser = argparse.ArgumentParser(
    prog='yolo_renamer.py',
    description='Yolo file name renamer.',
)

parser.add_argument('-i', '--id', help='Class ID to count.',
                    dest='id', required=True)
args = parser.parse_args()

counter = 0

for i in glob.glob('./*.txt'):
    if (i == '.\classes.txt'):
        continue
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        if (data[0] == str(args.id)):
            counter += 1
            continue

print(f'{counter} object found.')