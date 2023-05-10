import os
import glob

jpgs = glob.glob('./*.jpg')
txts = glob.glob('./*.txt')

silinecekler = []

for i in jpgs:
    try:
        txts.index(i[:-3] + 'txt')
    except ValueError:
        print('Cannot find ' + i[:-3] + 'txt!')
        silinecekler.append(i)
for i in txts:
    try:
        jpgs.index(i[:-3] + 'jpg')
    except ValueError:
        print('Cannot find ' + i[:-3] + 'jpg!')
        silinecekler.append(i)

if(str(input('Delete files to remove missing data? (y/n): ')) == 'y'):
    for i in silinecekler:
        os.remove(i)