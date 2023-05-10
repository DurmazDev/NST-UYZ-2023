import os
import argparse
import random
from collections import Counter

# Forked from https://github.com/GotG/yolotinyv3_medmask_demo/
parser = argparse.ArgumentParser()
parser.add_argument('train_pct',help="percentage of images to be used for training, the rest are used for validation and testing",
                    type=int)
parser.add_argument('valid_pct',help="percentage of images to be used for validation, the rest are used for training and testing",
                    type=int)
parser.add_argument('yolodataset',help="path to directory with images and yolo annotations",
                    type=str)
args = parser.parse_args()


counter = 1  

extensions = []
images=[]
#check extension of files in folder:
for filename in os.scandir(args.yolodataset):
    title,ext = os.path.splitext(filename.name)
    if ext != ".txt":
       extensions.append(ext)
       
ext_dict=Counter(extensions)
extension = max(ext_dict,key=ext_dict.get)       
print("Your image file extension is: " + extension)

for filename in os.listdir(args.yolodataset):  
    _, ext = os.path.splitext(os.path.basename(filename))
    if ext == extension:
       images.append(os.path.join(args.yolodataset,filename))

number_of_images = len(images)
   
index_valid = round(number_of_images*args.valid_pct/100)
validfiles = random.sample(images,index_valid)
traintestfiles = list(set(images).difference(set(validfiles)))
index_train = round(len(traintestfiles)*(args.train_pct/(100-args.valid_pct)))
trainfiles = random.sample(traintestfiles,index_train)
testfiles = list(set(traintestfiles).difference(set(trainfiles)))

print('Number of images:',number_of_images)


with open('train.txt', mode='w') as f:
    for item in trainfiles:
        f.write(item + "\n")

with open('valid.txt', mode='w') as f:
    for item in validfiles:
        f.write(item + "\n")
with open('test.txt', mode='w') as f:
    for item in testfiles:
        f.write(item + "\n")
print('Number of images used for training',str(len(trainfiles)))
print('Number of images used for validation',str(len(validfiles)))
print('Number of images used for testing',str(len(testfiles)))
