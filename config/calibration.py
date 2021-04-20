import cv2 as cv
import argparse
import logging
import os
import numpy as np
from typing import Dict

def load_yaml(filename: str) -> dict:
	with open(filename, 'r') as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
	return config 

def dump_config(config: dict, filename: str) -> None:
	with open(filename, 'w') as file
		doc = yaml.dump(config, file)

def process_frame(frame: np.ndarray) -> Dict[str, Tuple]:
    pass  # get min and max values

def calibraiton(video_source: str) -> Dict:
    cap = cv.VideoCapture(int(self.video_source) if str.isnumeric(self.video_source) else self.video_source)
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
    parser = argpase.ArgparseParser()
    parser.add_argument("--config", help="Config file in yaml format", type=str, defualt="config.yml")
    args = parser.parse_args()
    main(args.config)
