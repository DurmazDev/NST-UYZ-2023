import glob
import os
import argparse

parser = argparse.ArgumentParser(
    prog='delete_useless_files.py',
    description='Program to delete useless label/image files in training data. Be careful using this program. It can be damage data.',
)

parser.add_argument('-i', '--images', help='Images directory.',
                    dest='images_dir', required=True)
parser.add_argument('-l', '--labels', help='Labels directory.',
                    dest='labels_dir', required=True)
parser.add_argument('-e', '--ext', required=False,
                    dest='extension', help='Image extension.', default='jpg')

args = parser.parse_args()
deleted_counter = 0

for i in glob.glob(args.labels_dir + '*.txt'):
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    if (len(lines) > 0):
        continue
    image_name = i.replace(args.images_dir, '')[:-3] + args.extension
    os.remove(args.images_dir + image_name)
    os.remove(i)
    deleted_counter += 1

print(f"Deleted item count: {deleted_counter}")
