import glob
import os
import argparse

parser = argparse.ArgumentParser(
    prog='delete_classname.py',
    description='Scans the YOLO labels in the folder and deletes negative values.',
)

parser.add_argument('-d', '--dir', help='Directory of yolo labels', dest='dir', required=True)
args = parser.parse_args()

for i in glob.glob(str(args.dir) + '/*.txt'):
    if (i == '.\classes.txt'):
        continue
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    tempLines = []
    for j in lines:
        trigger = False
        j = j.replace('\n', '')
        data = j.split(' ')
        for l in range(len(data)):
            if (l == 0):
                continue
            if (float(data[l]) < 0 or float(data[l]) > 1):
                trigger = True
                break
        if (trigger == True):
            continue
        str1 = ""
        for k in data:
            str1 += k + " "
        tempLines.append(str1[:-1] + "\n")
    if (len(tempLines) != 0):
        f = open(i, 'w')
        f.writelines(tempLines)
        f.close()
    else:
        os.remove(i)
        try:
            os.remove(i[:-3] + 'jpg')
        except:
            continue
