"""
main.py

Specific implementation of the Dispatcher class to recognize and track people inside an indoor environment.
Stats about paths followed by people, social interaction and an heatmap of the most crowded areas will be available soon
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import argparse
import logging
import os
import numpy as np
import time
from typing import Tuple, List
from enum import Enum, auto
from math import floor, ceil
from cvlib import *
from config import config_boundarys
from dispatcher import Dispatcher

def main(video_source, image: str, models_name: str, processed_frame_preview: bool) -> None:
    models = list()
    for model_name in models_name:
        models.append(os.path.join(os.path.split(os.path.abspath(cv.__file__))[0], 'data', model_name))
        
    disp = Dispatcher(models, video_source = video_source, image = image)
    disp.loop(processed_frame_preview)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', help='List of cascade classifier model names. Path relative to cv2 install dir', default=['haarcascade_frontalface_default.xml'], nargs='+')
    parser.add_argument('--source', help='Camera number or video filename', type=str, default='0')
    parser.add_argument('--image', help='Image filename', type=str)
    parser.add_argument('--processed-frame-preview', help='Show the preview of processed frame', default=False, action='store_true')
    args = parser.parse_args()
    main(args.source, args.image, args.models, args.processed_frame_preview)
