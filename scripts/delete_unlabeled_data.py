import glob
import os
import argparse

parser = argparse.ArgumentParser(
    prog='delete_unlabeled_data.py',
    description='Scans the YOLO labels in the folder and deletes unchecked values.',
)

parser.add_argument('-i', '--images', help='Images directory.',
                    dest='images_dir', required=True)
parser.add_argument('-l', '--labels', help='Labels directory.',
                    dest='labels_dir', required=True)
parser.add_argument('-e', '--ext', required=False,
                    dest='extension', help='Image extension.', default='jpg')

args = parser.parse_args()
deleted_counter = 0

for i in glob.glob(args.images_dir + f'*.{args.extension}'):
    try:
        f = open(args.labels_dir + i.replace(args.images_dir, '')
                 [:-3] + 'txt').close()
    except FileNotFoundError:
        os.remove(i)
        deleted_counter += 1

print(deleted_counter + ' data deleted.')