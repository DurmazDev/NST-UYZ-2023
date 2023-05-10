import os
import multiprocessing
import time


def train_mover(i):
    i = i.replace('\n', '')
    os.system(f'mv {i} images/train')
    os.system(f'mv {i[:-3]}txt labels/train')
def test_mover(i):
    i = i.replace('\n', '')
    os.system(f'mv {i} images/test')
    os.system(f'mv {i[:-3]}txt labels/test')
def val_mover(i):
    i = i.replace('\n', '')
    os.system(f'mv {i} images/val')
    os.system(f'mv {i[:-3]}txt labels/val')


start = time.time()
if __name__ == '__main__':
    os.mkdir('images')
    os.mkdir('images/train/')
    os.mkdir('images/test')
    os.mkdir('images/val')
    os.mkdir('labels')
    os.mkdir('labels/train/')
    os.mkdir('labels/test')
    os.mkdir('labels/val')

    f = open('train.txt')
    train_lines = f.readlines()
    f.close()
    f = open('test.txt')
    test_lines = f.readlines()
    f.close()
    f = open('valid.txt')
    val_lines = f.readlines()
    f.close()

    pool = multiprocessing.Pool()
    pool.map(train_mover, train_lines)
    pool.close()
    pool.join()
    print('Moving train finished in ' +
          str(time.time() - start) + ' seconds.')
    pool = multiprocessing.Pool()
    pool.map(test_mover, test_lines)
    pool.close()
    pool.join()
    print('Moving test finished in ' +
          str(time.time() - start) + ' seconds.')
    pool = multiprocessing.Pool()
    pool.map(val_mover, val_lines)
    pool.close()
    pool.join()
    print('Moving val finished in ' +
          str(time.time() - start) + ' seconds.')
