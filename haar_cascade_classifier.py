import cv2 as cv
import argparse
import logging
import os

class Point:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x: int = x
        self.y: int = y

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
        return Point(x + w // 2, y + h // 2)

class Classifier:
    def __init__(self, video_source: int, model_name: str) -> None:
        """
        model_name: relative path to the xml model
        """
        self.model_cascade = cv.CascadeClassifier()
        self.model_cascade.load(cv.samples.findFile(model_name))
        self.video_source = video_source

    @static
    def draw_ellipse(frame, region: Region):
        return cv.ellipse(frame, region.get_center(), (w // 2, h // 2), 0, 0, 360, (0, 255, 0), 4)

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
            frame = draw_ellipse(frame, region)
        cv.imshow('Face detection with HCC', frame)  # HCC - Haar Cascade Classifier

    def detect_and_display(self, frame) -> None:
        pass


def main(video_camera: int, cascade_classifier_name: str) -> None:
    classifier = Classifier(video_source, cascade_classifier_name)
    cap = cv.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error("Camera video stream can't be opened")
        exit(1)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No caputered frame')
            continue

        detect_and_display(frame, face_cascade)

        if cv.waitKey(10) == 27:
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_cascade', help='Path to face cascade.',
                        default=os.path.join(os.path.split(os.path.abspath(cv.__file__))[0],
                                             'data', 'haarcascade_frontalface_alt.xml'))
    parser.add_argument('--camera', help='Camera number.', type=int, default=0)
    args = parser.parse_args()
    fc = args.face_cascade
    c = args.camera
    main(c, fc)
