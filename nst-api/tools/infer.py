#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import torch
import json
import concurrent.futures
import logging
import json
import requests
import time
import cv2
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from decouple import config

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer

from kask import KASK
from constants import *
from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6s.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')
    parser.add_argument('--continue-from', type=str, default='', help='Continue from frame')

    args = parser.parse_args()
    LOGGER.info(args)
    return args

def download_image(img_url, images_folder):
    t1 = time.perf_counter()
    img_bytes = requests.get(img_url).content
    image_name = img_url.split("/")[-1]
    with open(images_folder + image_name, 'wb') as img_file:
        img_file.write(img_bytes)
    t2 = time.perf_counter()
    logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
    return images_folder + image_name

@torch.no_grad()
def run(
        weights=osp.join(ROOT, 'yolov6s.pt'),
        source=osp.join(ROOT, 'data/images'),
        webcam=False,
        webcam_addr=0,
        yaml=None,
        img_size=640,
        conf_thres=0.3,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=False,
        not_save_img=True,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='nst',
        hide_labels=False,
        hide_conf=False,
        half=False,
        continue_from='',
        ):
    print("Başlatılıyor...")

    # Create save dir
    if save_dir is None:
        save_dir = osp.join(project, name)
        save_txt_path = osp.join(save_dir, 'labels')
    else:
        save_txt_path = save_dir
    if (not not_save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning('Save directory already existed.')
    
    # WARN(Ahmet): Burada kesilecek pixel değerleri sliding-window tarafından hesaplanmakta
    # fakat yarışmada değerler statik olarak 1920x1080 geleceği için tekrar tekrar çalıştırıp
    # performans kaybından kaçındık. Bu yüzden değerler statik olarak constants.py içerisine atanmıştır.
    inferer = Inferer("./cropped/", webcam, webcam_addr, weights, device, yaml, img_size, half)

    # Get configurations from .env file
    config.search_path = "./configs/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    sended_prediction_count = 0
    start_time = time.time()
    
    # Initialize KASK
    kask = KASK(custom_crop_pixels, custom_conf)

    trigger = True # Set False for continue from X frame.
    # Run object detection model frame by frame.
    for frame in tqdm(frames_json):
        # Belirli bir resimden devam etmek için continue_from parametresini kullan.
        if(continue_from not in frame["image_url"] and trigger == False):
            continue
        else:
            trigger = True
        if (time.time() - start_time < 60.0 and sended_prediction_count >= 75):
            print(f"Limite yaklaşıldı! {60 - (time.time() - start_time)} saniye beklenecek!!")
            time.sleep(60 - (time.time() - start_time))
            sended_prediction_count = 0
            start_time = time.time()

        # Create a prediction object to store frame info and detections
        prediction_start = time.time()
        frame_id = int(frame["url"].split('/')[-2])
        user_data = evaluation_server_url + 'users/XXX/'

        frame_preds = FramePredictions(frame['url'], frame['image_url'], frame['video_name'], frame_id, user_data)

        # Download image to _images.
        downloaded_image_path = download_image(evaluation_server_url + "media" + frame_preds.image_url, "./_images/")
        image = cv2.imread(downloaded_image_path)
        if(kask.check_blur(image)):
            LOGGER.info("Blur tespit edildi. Noise uygulanıyor...")
            image = kask.add_noise(image, downloaded_image_path)
        kask.cutter(image, "./cropped/")

        # Run detection model
        inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img)
        unified_predictions = kask.unifier(inferer.predictions)
        clear_predictions = kask.bug_fixer(unified_predictions, 45)
        kask.add_teknofest_detected(clear_predictions, frame_preds)
        inferer.clear_predictions()

        prediction_time = time.time() - prediction_start
        LOGGER.info(frame["url"] + " predicted in " + str(prediction_time) + "seconds.")

        # Send model predictions of this frame to the evaluation server
        while True:
            result = server.send_prediction(frame_preds)
            if result.status_code == 201:
                logging.info(f'Successfully sended. Download & Prediction time: {str(prediction_time)}, POST request time: {str(result.elapsed.total_seconds())}')
                sended_prediction_count += 1
                break
            else:
                print("ERROR!! \n\t{}".format(result.text))
                try:
                    response_json = json.loads(result.text)
                    if "You do not have permission to perform this action." in response_json["detail"]:
                        print("Limit exceeded. 80frames/min \n\t{}".format(result.text))
                        time.sleep(1)
                    if "You have already send prediction for this frame" in response_json["detail"]:
                        break
                except:
                    print(result)
                    print("UNKNOWN ERROR OCCURRED. PROGRAM STOPPED.")
                    sys.exit()

def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
