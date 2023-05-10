import multiprocessing
import cv2
import glob
import time
import os


def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    return (xmin, ymin, xmax, ymax)


def multip(i):
    classes = ['otomobil', 'motosiklet', 'otobus', 'kamyon', 'gemi',
               'insan', 'uap', 'uai', 'kepce', 'tren', 'vagon', 'yuk_gemisi']
    class_colors = [(83, 232, 101), (127, 217, 105), (181, 80, 15), (35, 235, 32), (49, 146, 122), (219, 128, 55),
                    (141, 19, 217), (230, 136, 195), (250, 18, 240), (216, 96, 247), (161, 193, 177), (138, 123, 134)]
    img = cv2.imread(i[:-3] + 'jpg')
    height, width = img.shape[:2]
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        xmin, ymin, xmax, ymax = unconvert(width, height, float(data[1]), float(
            data[2]), float(data[3]), float(data[4]))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      class_colors[int(data[0])], 2)
        cv2.putText(img, classes[int(data[0])], (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, class_colors[int(data[0])], 2)
    cv2.imwrite('out/' + i[:-3] + 'jpg', img)


start = time.time()
if __name__ == '__main__':
    if (not os.path.isdir('out')):
        os.mkdir('out')
    pool = multiprocessing.Pool()
    pool.map(multip, glob.glob('*.txt'))
    pool.close()
    pool.join()
    print('Painting done in ' + str(time.time() - start) + ' seconds.')
