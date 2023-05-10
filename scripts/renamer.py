import os
import multiprocessing
import uuid
import time
import glob


def convert_images(i):
    name = str(uuid.uuid4())
    os.system(f'move "{i}" "{name}.jpg"')
    os.system(f'move "{i[:-3]}txt" "{name}.txt"')


start = time.time()
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool.map(convert_images, glob.glob('./*.jpg'))
    pool.close()
    pool.join()
    print('Renaming images finished in ' +
          str(time.time() - start) + ' seconds.')
