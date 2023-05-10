import cv2
import os
import random
import numpy as np
import multiprocessing
import glob
import time
from tqdm import tqdm


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def unconvert(width, height, x, y, w, h):
    xmin = int((x - w / 2) * width)
    xmax = int((x + w / 2) * width)
    ymin = int((y - h / 2) * height)
    ymax = int((y + h / 2) * height)
    return (xmin, ymin, xmax, ymax)


def check_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = float(box1[0]), float(
        box1[1]), float(box1[2]), float(box1[3])
    x2_min, y2_min, x2_max, y2_max = float(box2[0]), float(
        box2[1]), float(box2[2]), float(box2[3])
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return x_overlap > 0 and y_overlap > 0


def sliding_view(image, window_size, stride, filename):
    # image: işlem yapılacak görüntü
    # window_size: kaydırma penceresinin boyutu (genişlik, yükseklik)
    # stride: pencerenin kaydırma adımı (genişlik, yükseklik)
    height, width, channel = image.shape

    window_width, window_height = window_size
    stride_width, stride_height = stride

    for y in range(0, height, stride_height):
        for x in range(0, width, stride_width):
            x1 = x
            y1 = y
            x2 = x + window_width
            y2 = y + window_height
            if (x + window_width > width):
                x1 = width - window_size[0]
                x2 = width
            if (y + window_height > height):
                y1 = height - window_size[1]
                y2 = height
            yield [(x1, y1), (x2, y2)]


def multip(image_name):
    # Resim dosyasını yükle
    image = cv2.imread(image_name)
    default_height, default_width = image.shape[:2]

    # Kaydedilecek bölüm boyutu (genişlik, yükseklik)
    window_size = (640, 640)

    # Kaydırma adımı (genişlik, yükseklik)
    stride = (512, 512)
    inner_counter = 0

    if (default_height < window_size[0]):
        zeros = np.zeros(
            (window_size[0], default_width, image.shape[2]), dtype=int)
        zeros[:default_height, :, :] = image
        image = zeros

    crop_sizes = []
    for i, window in enumerate(sliding_view(image, window_size, stride, image_name)):
        filename = os.path.join(
            output_dir, f"{image_name[:-4]}_{inner_counter}.jpg")
        trigger = False
        try:
            crop_sizes.index(window)
            trigger = True
        except:
            crop_sizes.append(window)
        if (trigger):
            continue
        from_x, from_y = window[0]
        to_x, to_y = window[1]
        cropped_image = image[from_y:to_y, from_x:to_x]
        cv2.imwrite(filename, cropped_image)
        inner_counter += 1

    f = open(image_name[:-3] + 'txt', 'r')
    lines = f.readlines()
    f.close()

    for j in lines:
        j = j.replace('\n', '')
        data = j.split(' ')
        try:
            xmin, ymin, xmax, ymax = unconvert(default_width, default_height, float(data[1]), float(
                data[2]), float(data[3]), float(data[4]))
        except:
            os.system(f'del {image_name}')
            os.system(f'del {image_name[:-3]}txt')
            continue
        for k in range(len(crop_sizes)):
            overlap = check_overlap((xmin, ymin, xmax, ymax),
                                    (crop_sizes[k][0][0], crop_sizes[k][0][1],
                                        crop_sizes[k][1][0], crop_sizes[k][1][1]))
            if (not overlap):
                continue
            new_width, new_height = window_size
            t_xmax = xmax - crop_sizes[k][0][0]
            if (t_xmax < 0):
                t_xmax = 0
            elif (t_xmax > window_size[0]):
                t_xmax = window_size[0]
            t_xmin = xmin - crop_sizes[k][0][0]
            if (t_xmin < 0):
                t_xmin = 0
            elif (t_xmin > window_size[0]):
                t_xmin = window_size[0]
            t_ymax = ymax - crop_sizes[k][0][1]
            if (t_ymax < 0):
                t_ymax = 0
            elif (t_ymax > window_size[0]):
                t_ymax = window_size[0]
            t_ymin = ymin - crop_sizes[k][0][1]
            if (t_ymin < 0):
                t_ymin = 0
            elif (t_ymin > window_size[0]):
                t_ymin = window_size[0]
            if ((t_xmax - t_xmin) * (t_ymax - t_ymin) < 1000 and data[0] not in ['5', '1']):
                continue
            x, y, w, h = convert(
                (new_width, new_height),
                (t_xmin, t_xmax, t_ymin, t_ymax)
            )
            f = open(f'bolumler/{image_name[:-4]}_{k}.txt', 'a')
            f.write(f'{data[0]} {x} {y} {w} {h}\n')
            f.close()


def delete_unlabeled():
    deleted_counter = 0
    image_data = glob.glob(f'./{output_dir}/*.jpg')
    text_data = glob.glob(f'./{output_dir}/*.txt')
    for i in image_data:
        try:
            if (text_data.index(i[:-3] + 'txt') == -1):
                os.remove(i)
                deleted_counter += 1
        except:
            os.remove(i)
            deleted_counter += 1
    print(deleted_counter, 'items deleted.')


def suspicious_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    if (len(lines) == 1):
        j = lines[0]
        j = j.replace('\n', '')
        data = j.split(' ')
        # Burada bulunan 832 window_size değeridir. Atama yapmaya üşendim...
        xmin, ymin, xmax, ymax = unconvert(640, 640, float(data[1]), float(
            data[2]), float(data[3]), float(data[4]))
        if ((xmax - xmin) * (ymax - ymin) < 1000):
            os.system(f'move {filename} suspicious/')
            os.system(f'move {filename[:-3]}jpg suspicious/')


output_dir = 'bolumler'
if __name__ == '__main__':
    # Kaydedilecek klasörü oluştur
    start = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists('suspicious'):
        os.makedirs('suspicious')

    pool = multiprocessing.Pool()
    num_files = len(glob.glob('./*.jpg'))
    with tqdm(total=num_files) as pbar:
        for _ in pool.imap_unordered(multip, glob.glob('./*.jpg')):
            pbar.update()

    pool.close()
    pool.join()
    delete_unlabeled()
    print('Cropping images finished in ' +
          str(time.time() - start) + ' seconds. \n Scanning for suspicious data...')
    pool = multiprocessing.Pool()
    num_files = len(glob.glob(f'./{output_dir}/*.txt'))
    with tqdm(total=num_files) as pbar:
        for _ in pool.imap_unordered(suspicious_data, glob.glob(f'./{output_dir}/*.txt')):
            pbar.update()

    pool.close()
    pool.join()
    print('Suspicious data scanning completed. Check suspicious data manually. If there is no error on data, move images and label failes to main dataset folder.')
