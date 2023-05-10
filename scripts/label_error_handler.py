import glob
import os
import multiprocessing


def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    return (xmin, ymin, xmax, ymax)


def to_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert(i):
    f = open(i, 'r')
    lines = f.readlines()
    f.close()
    new_lines = []
    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        d1 = float(data[1])
        d2 = float(data[2])
        d3 = float(data[3])
        d4 = float(data[4])
        if (d1 < 0.0 or d1 > 1.0 or d2 < 0.0 or d2 > 1.0 or d3 < 0.0 or d3 > 1.0 or d4 < 0.0 or d4 > 1.0):
            xmin, ymin, xmax, ymax = unconvert(640, 640, float(
                data[1]), float(data[2]), float(data[3]), float(data[4]))
            print('[')
            print(i)
            print(xmin, ymin, xmax, ymax)
            if (xmin < 0):
                xmin = 0
            if (xmax < 0):
                xmax = 0
            if (ymin < 0):
                ymin = 0
            if (ymax < 0):
                ymax = 0
            if (xmin > 640):
                xmin = 640
            if (xmax > 640):
                xmax = 640
            if (ymin > 640):
                ymin = 640
            if (ymax > 640):
                ymax = 640
            yolo_data = to_yolo((640, 640), (xmin, xmax, ymin, ymax))
            print(xmin, ymin, xmax, ymax)
            print(yolo_data)
            print(']')
            new_lines.append(
                f'{data[0]} {yolo_data[0]} {yolo_data[1]} {yolo_data[2]} {yolo_data[3]}\n')
        else:
            new_lines.append(
                f'{data[0]} {data[1]} {data[2]} {data[3]} {data[4]}\n')
    f = open(i, 'w')
    f.writelines(new_lines)
    f.close()


if __name__ == '__main__':
    dirs = os.listdir('./')
    for i in dirs:
        if (not os.path.isdir(i)):
            continue
        pool = multiprocessing.Pool()
        pool.map(convert, glob.glob(f'{i}/*.txt'))
        pool.close()
        pool.join()
