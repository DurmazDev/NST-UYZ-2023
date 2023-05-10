import os
import glob
import cv2
import multiprocessing
import time


def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    return ((xmax - xmin) * (ymax - ymin))


def convert(i):
    if (i == 'classes.txt'):
        return
    img = cv2.imread(i[:-3] + 'jpg')
    shape = img.shape[:2]
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    new_lines = []
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        m2 = unconvert(shape[1], shape[0],
                       float(data[1]), float(data[2]), float(data[3]), float(data[4]))
        if (m2 < 100):
            continue
        new_lines.append(j + '\n')
    if (len(new_lines) == 0):
        os.system(f'del {i}')
        os.system(f'del {i[:-3]}jpg')
    else:
        f = open(i, 'w')
        f.writelines(new_lines)
        f.close()


start = time.time()
if __name__ == '__main__':
    files = glob.glob('suspicious\\*.txt')
    pool = multiprocessing.Pool()
    pool.map(convert, files)
    pool.close()
    pool.join()
    print('Converting finished in ' +
          str(time.time() - start) + ' seconds.')
