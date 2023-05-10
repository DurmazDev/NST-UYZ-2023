import os
import glob


def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    pbar = tqdm(glob.glob('annotations/*.txt'))
    for f in pbar:
        image_name = 'images/' + f.replace('annotations\\', '')[:-3] + 'jpg'
        img_size = Image.open(image_name).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                if cls == 0:
                    cls = 5
                elif cls == 1:
                    cls = 5
                elif cls == 2:
                    continue
                elif cls == 3:
                    cls = 0
                elif cls == 4:
                    cls = 0
                elif cls == 5:
                    cls = 3
                elif cls == 6:
                    continue
                elif cls == 7:
                    continue
                elif cls == 8:
                    cls = 2
                elif cls == 9:
                    continue
                elif cls == 10:
                    continue
                else:
                    continue
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt


visdrone2yolo(".")
