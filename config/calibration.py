import cv2 as cv
import argparse
import logging
import os
import numpy as np
from typing import Dict, Tuple
import yaml
from ymlparser import *

import sys
sys.path.append('..')
from cvlib.display import Display
from cvlib.utils import Shape, Region

face_boundarys_max = (0, 0)  # These values will be overwritten by process_frame method: (w, h)
body_boundarys_max = (0, 0)  # These values will be overwritten by process_frame method: (w, h)
face_boundarys_min = (float('inf'), float('inf'))  # These values will be overwritten by process_frame method: (w, h)
body_boundarys_min = (float('inf'), float('inf'))  # These values will be overwritten by process_frame method: (w, h)
model_path = "../venv/lib/python3.8/site-packages/cv2/data/"
face_model = "haarcascade_frontalface_default.xml"
body_model = "haarcascade_fullbody.xml"
face = cv.CascadeClassifier()
body = cv.CascadeClassifier()
d = Display()
first_frame = True


def preprocess(frame: np.ndarray) -> None:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame, dsize = d.size, interpolation = cv.INTER_AREA)
    # frame = cv.equalizeHist(frame)
    return frame

def process_frame(frame: np.ndarray) -> Dict[str, Tuple]:
    global face, body, face_boundarys_max, body_boundarys_max, face_boundarys_min, body_boundarys_min, d, first_frame
    if first_frame:
        d.set_orientation(frame)
        first_frame = False
    frame = preprocess(frame)
    detected_faces = face.detectMultiScale(frame, scaleFactor=1.1, minSize=(5, 5))  # maxSize is equal to the size of the frame by default
    detected_bodys = body.detectMultiScale(frame, scaleFactor=1.1, minSize=(5, 5))  # maxSize is equal to the size of the frame by default
    roi: List[Region] = list()  # List of regions
    print(detected_faces)
    for f in detected_faces:
        roi.append(Region(*f, (255, 0, 0), Shape.CIRCLE))
        face_boundarys_max = max(face_boundarys_max, (f[2], f[3]))
        face_boundarys_min = min(face_boundarys_min, (f[2], f[3]))
    logging.info(f"Max face until now: {face_boundarys_max}")
    logging.info(f"Min face until now: {face_boundarys_min}")
    for b in detected_bodys:
        roi.append(Region(*b, (0, 0, 255), Shape.RECTANGLE))
        body_boundarys_max = max(body_boundarys_max, (b[2], b[3]))
        body_boundarys_min = min(body_boundarys_min, (b[2], b[3]))
    logging.info(f"Max body until now: {body_boundarys_max}")
    logging.info(f"Min body until now: {body_boundarys_min}")
    d.show(frame, roi, "Calibration")
    

def calibration(video_source: str) -> Dict:
    global face, body, face_model, body_model
    # Loading models
    face_model = os.path.join(model_path, face_model)
    body_model = os.path.join(model_path, body_model)
    face.load(cv.samples.findFile(face_model))
    body.load(cv.samples.findFile(body_model))
    cap = cv.VideoCapture(int(video_source) if str.isnumeric(video_source) else video_source)
    frames_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    is_file: bool = True if frames_num > 0 else False
    if is_file:
        step = int(frames_num*0.05)
        current_frame = 0
    if not cap.isOpened():
        logging.error("Camera video stream can't be opened")
        exit(1)
    while True:
        if is_file:
            current_frame += step
            cap.set(1, current_frame)
        ret, frame = cap.read()
        if frame is None:
            break
        process_frame(frame)
        if cv.waitKey(1) == 27:  # Key ==> 'ESC'
            break

def main(source: str, config_file: str) -> None:
    global body_boundarys_min, body_boundarys_max, face_boundarys_min, face_boundarys_max
    config = dict()
    config = load_yaml(config_file)
    calibration(source)
    config['min']['body']['width'] = str(body_boundarys_min[0])
    config['min']['body']['height'] = str(body_boundarys_min[1])
    config['max']['body']['height'] = str(body_boundarys_max[0])
    config['max']['body']['width'] = str(body_boundarys_max[1])
    config['min']['face']['width'] = str(face_boundarys_min[0])
    config['min']['face']['height'] = str(face_boundarys_min[1])
    config['max']['face']['height'] = str(face_boundarys_max[0])
    config['max']['face']['width'] = str(face_boundarys_max[1])
    dump_config(config, config_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file in yaml format", type=str, default="config.yml")
    parser.add_argument("--source", help="Source stream. Either a file or a webcam", type=str, default="0")
    args = parser.parse_args()
    main(args.source, args.config)
