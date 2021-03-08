import cv2 as cv
import argparse
import logging
import os
import numpy
import time
from typing import Tuple, List
from enum import Enum, auto

VGA_HORIZONTAL_SIZE: Tuple[int, int] = (640, 480)
VGA_VERTICAL_SIZE: Tuple[int, int] = tuple(reversed(VGA_HORIZONTAL_SIZE))

class Point:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x: int = x
        self.y: int = y

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

class Region:
    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> None:
        """
        x, y: coordinates of upper left point
        w, h: size of the region
        """
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h

    def get_area(self) -> int:
        return (self.w * self.h)

    def get_center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

class Orientation(Enum):
    VERTICAL: int = auto()
    HORIZONTAL: int = auto()
    SQUARE: int = auto()

    @staticmethod
    def get_orientation(img: numpy.ndarray):
        w: int = img.shape[1]
        h: int = img.shape[0]
        if w > h:
            return Orientation.HORIZONTAL
        elif h > w:
            return Orientation.VERTICAL
        else:
            return Orientation.SQUARE

class Classifier:
    def __init__(self, video_source: str, model_name: str) -> None:
        """
        model_name: relative path to the xml model
        """
        self.model_cascade: cv.CascadeClassifier = cv.CascadeClassifier()
        self.model_cascade.load(cv.samples.findFile(model_name))
        self.video_source: int = video_source
    
    def get_time(self) -> int:
        return time.time()

    def draw_ellipse(self, frame: numpy.ndarray, region: Region) -> numpy.ndarray:
        return cv.ellipse(frame, region.get_center().to_tuple(), (region.w // 2, region.h // 2), 0, 0, 360, (0, 255, 0), 4)
    
    def detect(self, frame: numpy.ndarray, processed_frame_preview: bool = False) -> List[Region]:
        start: int = self.get_time()
        frame_gray: numpy.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray: numpy.ndarray = cv.equalizeHist(frame_gray)
        orientation: Orientation = Orientation.get_orientation(frame)
        if orientation is Orientation.VERTICAL:
            size: Tuple[int, int] = VGA_VERTICAL_SIZE
        elif orientation is Orientation.HORIZONTAL:
            size: Tuple[int, int] = VGA_HORIZONTAL_SIZE
        downscaled_frame_gray: numpy.ndarray = cv.resize(frame_gray, dsize = size, interpolation = cv.INTER_AREA)
        obj_list = self.model_cascade.detectMultiScale(downscaled_frame_gray)
        logging.info(f"time for 1 frame classification {self.get_time() - start}")
        l: List[Region] = list()
        for (x, y, w, h) in obj_list:
            l.append(Region(x, y, w, h))
        if processed_frame_preview:
            self.display(downscaled_frame_gray, l, 'Processed frame preview')
        return l

    def display(self, frame: numpy.ndarray, regions: List[Region], window_title: str = 'OpenCV show', scale_factor: float = 1.0) -> None:
        for region in regions:
            frame: numpy.ndarray = self.draw_ellipse(frame, region)
        cv.imshow(window_title, scale(frame, scale_factor))  # HCC - Haar Cascade Classifier
    
    def detect_and_display(self, frame: numpy.ndarray, processed_frame_preview: bool) -> None:
        regions: List[Region] = self.detect(frame, processed_frame_preview)
        self.display(frame, regions, 'Face detection with HCC', 0.5)

    def start(self, processed_frame_preview: bool) -> None:  # Blocking method
        cap = cv.VideoCapture(int(self.video_source) if str.isnumeric(self.video_source) else self.video_source)
        if not cap.isOpened():
            logging.error("Camera video stream can't be opened")
            exit(1)
        while True:
            ret, frame = cap.read()
            if frame is None:
                continue
            self.detect_and_display(frame, processed_frame_preview)
            if cv.waitKey(10) == 27:  # Key ==> 'ESC'
                break

def scale(frame: numpy.ndarray, scale_factor: float) -> numpy.ndarray:  # scale_factor between 0 and 1 if you want to scale down the image
    scaled_h: int = int(frame.shape[0] * scale_factor)
    scaled_w: int = int(frame.shape[1] * scale_factor)
    return cv.resize(frame, (scaled_w, scaled_h))

def main(video_source: str, model: str, processed_frame_preview: bool) -> None:
    classifier = Classifier(video_source, model)
    classifier.start(processed_frame_preview)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to cascade classifier model.', default=os.path.join(os.path.split(os.path.abspath(cv.__file__))[0], 'data', 'haarcascade_frontalface_alt.xml'))
    parser.add_argument('--source', help='Camera number or video filename.', type=str, default='0')
    parser.add_argument('--processed-frame-preview', help='Show the preview of processed frame', default=False, action='store_true')
    args = parser.parse_args()
    main(args.source, args.model, args.processed_frame_preview)
