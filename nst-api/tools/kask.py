import cv2
import os
import numpy as np

from constants import *
from src.detected_object import DetectedObject

class KASK:
    def __init__(self, crop_pixels, custom_conf, second_model=False):
        self.custom_conf = custom_conf
        # crop_pixels, KASK'ın kesim yapacağı pixel aralıklarıdır. Bu işlemi normalde sliding_window metodu yapar.
        # İşlem hızı kazanma amacıyla bu kısım önceden ayarlanıp statik olarak atanmıştır.
        # Dilerseniz sliding_window metodunu bağlayıp otomatik piksel kesimi yapabilirsiniz.
        self.crop_pixels = crop_pixels
        self.second_model = second_model

    def check_overlap(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1["xmin"], box1["ymin"], box1["xmax"], box1["ymax"]
        x2_min, y2_min, x2_max, y2_max = box2["xmin"], box2["ymin"], box2["xmax"], box2["ymax"]
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        return x_overlap > 0 and y_overlap > 0

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA["xmin"], boxB["xmin"])
        yA = max(boxA["ymin"], boxB["ymin"])
        xB = min(boxA["xmax"], boxB["xmax"])
        yB = min(boxA["ymax"], boxB["ymax"])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (float(boxA["xmax"]) - float(boxA["xmin"]) + 1) * \
            (float(boxA["ymax"]) - float(boxA["ymin"]) + 1)
        boxBArea = (float(boxB["xmax"]) - float(boxB["xmin"]) + 1) * \
            (float(boxB["ymax"]) - float(boxB["ymin"]) + 1)
        iou = float(interArea / float(boxAArea + boxBArea - interArea))
        return iou

    def sliding_window(self, image, window_size, stride):
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

    def cutter(self, image, output_dir):
        inner_counter = 0
        for coords in self.crop_pixels:
            filename = os.path.join(output_dir, f"NSTPART_{inner_counter}.jpg")
            crop = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
            cv2.imwrite(filename, crop)
            inner_counter += 1

    def get_area(self, box):
        w = box["xmax"] - box["xmin"]
        h = box["ymax"] - box["ymin"]
        return float(w * h)

    def custom_conf_setter(self, predictions):
        delete_index = []
        for index, pred in enumerate(predictions):
            if (pred["conf"] < self.custom_conf[pred["class"]]):
                if(index not in delete_index):
                    delete_index.append(index)
        for i in sorted(delete_index, reverse=True):
            predictions.pop(i)
        return predictions

    def write_output(self, image_folder, image_name, output_folder, output_name, predictions):
        # DEVMODE(Ahmet): Bu kısım prod'da çalıştırılırsa yavaşlama olabilir.
        filename = os.path.join(image_folder, image_name)
        img = cv2.imread(filename)
        for i in predictions:
            img = cv2.rectangle(img, (i["xmin"], i["ymin"]), (i["xmax"], i["ymax"]), (0, 255, 0), 2)
            if(i["class"] in model_classes["Tasit"]):
                label = "Tasit" + str(i["conf"])
            elif (i["class"] == model_classes["Insan"]):
                label = "Insan" + str(i["conf"])
            elif (i["class"] == model_classes["UAP"]):
                label = "UAP" + str(i["conf"])
            elif (i["class"] == model_classes["UAI"]):
                label = "UAI" + str(i["conf"])
            cv2.putText(img, label, (i["xmin"], i["ymin"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        output_dir = os.path.join(output_folder, output_name)
        cv2.imwrite(output_dir, img)

    def check_blur(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = 100
        if laplacian_var < blur_threshold:
            return True
        return False

    def add_noise(self, img):
        mean = 0
        std_dev = 50
        shape = img.shape
        dtype = img.dtype
        noise = np.zeros(shape, dtype=dtype)
        cv2.randn(noise, mean, std_dev)
        noisy_img = cv2.add(img, noise)
        return noisy_img

    def bug_fixer(self, unified_predictions, max_iou):
        delete_index = []
        for i in range(len(unified_predictions)):
            for j in range(i + 1, len(unified_predictions)):
                if((unified_predictions[i]["class"] != model_classes["Insan"] or unified_predictions[j]["class"] != model_classes["Insan"]) and self.second_model == True):
                    continue
                overlap_value = self.check_overlap(unified_predictions[i], unified_predictions[j])
                if (overlap_value):
                    if((unified_predictions[i]["class"] == unified_predictions[j]["class"]) and unified_predictions[i]["class"] in [model_classes["UAP"], model_classes["UAI"]]):
                        if (self.get_area(unified_predictions[i]) > self.get_area(unified_predictions[j])):
                            if(j not in delete_index):
                                delete_index.append(j)
                        else:
                            if(i not in delete_index):
                                delete_index.append(i)
                        continue
                    iou = self.calculate_iou(unified_predictions[i], unified_predictions[j])
                    if (iou * 100 > max_iou):
                        if (self.get_area(unified_predictions[i]) > self.get_area(unified_predictions[j])):
                            if(j not in delete_index):
                                delete_index.append(j)
                        else:
                            if(i not in delete_index):
                                delete_index.append(i)
        for i in sorted(delete_index, reverse=True):
            unified_predictions.pop(i)
        unified_predictions = self.custom_conf_setter(unified_predictions)
        return unified_predictions
        

    def unifier(self, predictions):
        unified_predictions = []
        for pred in predictions:
            second_model = False
            if(pred["part"] not in [0, 1]):
                second_model = True
            # INFO(Ahmet): İkinci modeli sadece insanı tanıması için eğittik,
            # ikinci modelin 0. indexinde insan tanımlı bu yüzden otomatik olarak
            # insan verisi işaretlenerek kabul ediliyor.
            if(self.second_model):
                if(pred["class"] == 0):
                    pred["class"] = model_classes["Insan"]
                else:
                    continue
            if(second_model and pred["class"] != model_classes["Insan"]):
                continue
            elif((not second_model) and pred["class"] == model_classes["Insan"]):
                continue
            pred["xmin"] += self.crop_pixels[pred["part"]][0][0]
            pred["ymin"] += self.crop_pixels[pred["part"]][0][1]
            pred["xmax"] += self.crop_pixels[pred["part"]][0][0]
            pred["ymax"] += self.crop_pixels[pred["part"]][0][1]
            unified_predictions.append(pred)
        return unified_predictions
    
    def check_landable(self, predictions):
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                if (self.check_overlap(predictions[i], predictions[j])):
                    return landing_statuses["Inilemez"]
        return landing_statuses["Inilebilir"]
    
    # YOLOv6'dan çıkan işaretlemeleri TEKNOFEST yapısına çevir.
    def add_teknofest_detected(self, predictions, FramePredictions):
        for pred in predictions:
            print(pred)
            cls = ""
            landing_status = ""
            if(pred["class"] in model_classes["Tasit"]):
                cls = classes["Tasit"]
                landing_status = landing_statuses["Inis Alani Degil"]
            elif (pred["class"] == model_classes["Insan"]):
                cls = classes["Insan"]
                landing_status = landing_statuses["Inis Alani Degil"]
            elif (pred["class"] == model_classes["UAP"]):
                cls = classes["UAP"]
                landing_status = self.check_landable(predictions)
            elif (pred["class"] == model_classes["UAI"]):
                cls = classes["UAI"]
                landing_status = self.check_landable(predictions)

            detected_object = DetectedObject(
                cls,
                landing_status,
                pred["xmin"],
                pred["ymin"],
                pred["xmax"],
                pred["ymax"]
            )
            print(detected_object)
            FramePredictions.add_detected_object(detected_object)

        # WARN(Ahmet): Burası görüntüleme modu,
        # performansta sorun çıkarsa ilk olarak burası silinecek.
        frame_name = FramePredictions.image_url.split('/')[-1].replace('/', '')
        self.write_output('./_images/', frame_name, './_output/', frame_name[:-4] + '_out.jpg', predictions)