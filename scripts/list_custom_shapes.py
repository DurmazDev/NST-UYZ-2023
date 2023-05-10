import multiprocessing
import cv2
import glob
import time
import os

sizes = []


def convert_images(filename):
    global sizes
    img = cv2.imread(filename)
    shape = img.shape[:2]
    if (shape[0] < 640 or shape[1] < 640):
        os.system(f'del {filename}')
        os.system(f'del {filename[:-3]}txt')
        print(f'Deleted: {filename}')
    else:
        try:
            sizes.index(shape)
        except:
            sizes.append(shape)
            os.system(f'copy {filename} out\\')
            os.system(f'copy {filename[:-3]}txt out\\')
            print(f'Custom {str(shape)} shape found on {filename}')


start = time.time()
if __name__ == '__main__':
    os.mkdir('out')
    pool = multiprocessing.Pool()
    pool.map(convert_images, glob.glob('./*.jpg'))
    pool.close()
    pool.join()
    print(sizes)
