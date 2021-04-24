import cv2 as cv
import argparse
import logging
import os
import numpy as np
from typing import Dict, Tuple

import sys
sys.path.append('..')
from cvlib import display
from cvlib import utils

face_boundarys = (0, 0)  # These values will be overwritten by process_frame method
body_boundarys = (0, 0)  # These values will be overwritten by process_frame method
model_path = "../venv/lib/python3.8/site-packages/cv2/data/"
face_model = "haarcascade_frontalface_default.xml"
body_model = "haarcascade_fullbody.xml"
face = cv.CascadeClassifier()
body = cv.CascadeClassifier()
d = display.Display()

def load_yaml(filename: str) -> Dict:
    with open(filename, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    return config 

def dump_config(config: dict, filename: str) -> None:
    with open(filename, 'w') as file:
        doc = yaml.dump(config, file)

def process_frame(frame: np.ndarray) -> Dict[str, Tuple]:
    global face, body, face_boundarys, body_boundarys, d
    detected_faces = face.detectMultiScale(frame, scaleFactor=1.05, minSize=(5, 5))  # maxSize is equal to the size of the frame by default
    detected_bodys = body.detectMultiScale(frame, scaleFactor=1.05, minSize=(5, 5))  # maxSize is equal to the size of the frame by default
    for f in detected_faces:
        face_boundarys = max(face_boundarys, (f[0], f[1]))
    logging.info(f"Max face until now: {face_boundarys}")
    for b in detected_bodys:
        body_boundarys = max(body_boundarys, (b[0], b[1]))
    d.show(frame, detected_faces + detected_bodys, "Calibration")
    logging.info(f"Max body until now: {body_boundarys}")
    

def calibration(video_source: str) -> Dict:
    global face, body, face_model, body_model
    # Loading models
    face_model = os.path.join(model_path, face_model)
    body_model = os.path.join(model_path, body_model)
    face.load(cv.samples.findFile(face_model))
    body.load(cv.samples.findFile(body_model))
    cap = cv.VideoCapture(int(video_source) if str.isnumeric(video_source) else video_source)
    if not cap.isOpened():
        logging.error("Camera video stream can't be opened")
        exit(1)
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        process_frame(frame)
        if cv.waitKey(1) == 27:  # Key ==> 'ESC'
            break

def main(source: str, config_file: str) -> None:
    calibration(source)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file in yaml format", type=str, default="config.yml")
    parser.add_argument("--source", help="Source stream. Either a file or a webcam", type=str, default="0")
    args = parser.parse_args()
    main(args.source, args.config)
