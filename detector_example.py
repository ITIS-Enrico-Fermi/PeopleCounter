import cv2 as cv
from detector import Detector
import os
from sys import argv

class FrontalFaceDetector(Detector):
    def config(self):
        filename = self._model
        c = cv.CascadeClassifier()
        c.load(filename)
        self._detector = c

    def detect(self, frame):
        self._regions = self._detector.detectMultiScale(frame)
        return True

if __name__ == "__main__":
    if len(argv) == 2:  # The second parameter is the filename of the image to be tested against detectors
        d = FrontalFaceDetector.create() \
                               .set_model('OpencvModels/haarcascade_frontalface_default.xml') \
                               .run_config()
        d.detect(cv.imread(argv[1]))
        print(d.get_regions())
