import glob
import multiprocessing
import time
from PIL import Image


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


classes = []


def convert_data(i):
    if (i == 'classes.txt'):
        return
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        try:
            classes.index(data[0] + '\n')
        except ValueError:
            classes.append(data[0] + '\n')

    f = open('classes.txt', 'w')
    f.writelines(classes)
    f.close()

    for i in glob.glob('*.txt'):
        if (i == 'classes.txt'):
            continue
        f = open(i, 'r')
        lines = f.readlines()
        f.close()
        img = Image.open(i[:-3] + 'jpg')
        w, h = img.size
        file_yolo_data = []
        for j in lines:
            j = j.replace('\n', '')
            data = j.split(' ')
            object_class = classes.index(data[0] + '\n')
            yolo_str = ''
            out_data = convert(
                (w, h), (float(data[4]), float(data[6]),
                         float(data[5]), float(data[7])))
            yolo_str += str(object_class) + ' '
            yolo_str += f'{out_data[0]} {out_data[1]} {out_data[2]} {out_data[3]}\n'
            file_yolo_data.append(yolo_str)
        f = open(i, 'w')
        f.writelines(file_yolo_data)
        f.close()


if __name__ == '__main__':
    start = time.time()
    pool = multiprocessing.Pool()
    pool.map(convert, glob.glob('./*.txt'))
    pool.close()
    pool.join()
    print(time.time() - start)
