import cv2 as cv
import argparse
import logging
import os
from typing import Tuple, List

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

class Classifier:
    def __init__(self, video_source: str, model_name: str) -> None:
        """
        model_name: relative path to the xml model
        """
        self.model_cascade: cv.CascadeClassifier = cv.CascadeClassifier()
        self.model_cascade.load(cv.samples.findFile(model_name))
        self.video_source: int = video_source

    def draw_ellipse(self, frame, region: Region):
        return cv.ellipse(frame, region.get_center().to_tuple(), (region.w // 2, region.h // 2), 0, 0, 360, (0, 255, 0), 4)
    
    def detect(self, frame) -> List[Region]:
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        obj_list = self.model_cascade.detectMultiScale(frame_gray)
        l: List[Region] = list()
        for (x, y, w, h) in obj_list:
            l.append(Region(x, y, w, h))
        return l
    
    def display(self, frame, regions: List[Region]) -> None:
        for region in regions:
            frame = self.draw_ellipse(frame, region)
        cv.imshow('Face detection with HCC', cv.resize(frame, (800, 600)))  # HCC - Haar Cascade Classifier

    def detect_and_display(self, frame) -> None:
        regions: List[Region] = self.detect(frame)
        self.display(frame, regions)

    def start(self):
        cap = cv.VideoCapture(int(self.video_source) if str.isnumeric(self.video_source) else self.video_source)
        if not cap.isOpened():
            logging.error("Camera video stream can't be opened")
            exit(1)
        while True:
            ret, frame = cap.read()
            if frame is None:
                continue
            self.detect_and_display(frame)
            if cv.waitKey(10) == 27:  # Key ==> 'ESC'
                break


def main(video_source: str, model: str) -> None:
    classifier = Classifier(video_source, model)
    classifier.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to cascade classifier model.', default=os.path.join(os.path.split(os.path.abspath(cv.__file__))[0], 'data', 'haarcascade_frontalface_alt.xml'))
    parser.add_argument('--source', help='Camera number or video filename.', type=str, default='0')
    args = parser.parse_args()
    main(args.source, args.model)
