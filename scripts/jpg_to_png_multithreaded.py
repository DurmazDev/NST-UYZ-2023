import multiprocessing
from PIL import Image
import glob
import os
import time

start = time.time()


def convert(inpath):
    im1 = Image.open(inpath)
    im1.save(inpath[:-3] + 'png')
    os.remove(inpath)
    print(inpath)


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool.map(convert, glob.glob('./*.jpg'))
    pool.close()
    pool.join()
    print(time.time() - start)
