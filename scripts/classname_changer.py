import glob
import argparse

parser = argparse.ArgumentParser(
    prog='yolo_renamer.py',
    description='Yolo file name renamer.',
)

parser.add_argument('-f', '--from', help='From index',
                    dest='ffrom', required=True)
parser.add_argument('-t', '--to', help='To index',
                    dest='to', required=True)
args = parser.parse_args()

counter = 0

for i in glob.glob('./*.txt'):
    if (i == '.\classes.txt'):
        continue
    f = open(i, 'r')
    lines = f.readlines()
    f.close
    tempLines = []
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        if (data[0] == str(args.ffrom)):
            data[0] = str(args.to)
            counter += 1
        str1 = ""
        for k in data:
            str1 += k + " "
        tempLines.append(str1[:-1] + "\n")
    f = open(i, 'w')
    f.writelines(tempLines)
    f.close()
print(f'{counter} lines effected.')