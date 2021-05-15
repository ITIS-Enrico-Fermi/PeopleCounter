import os
import cv2 as cv
from sys import argv
from detector import DetectorMultiplexer, Detector

class OpencvDetector(Detector):
    def config(self):
        filename = self._model
        c = cv.CascadeClassifier()
        c.load(filename)
        self._detector = c

    def detect(self, frame):
        try:
            self._regions = self._detector.detectMultiScale(frame)
        except:
            return False
        return True

if __name__ == "__main__":
    if len(argv) == 2:  # The second parameter is the filename of the image to be tested against detectors
        frontal_face = OpencvDetector.create() \
                                     .set_model('OpencvModels/haarcascade_frontalface_default.xml') \
                                     .run_config()
        profile_face = OpencvDetector.create() \
                                     .set_model('OpencvModels/haarcascade_profileface.xml') \
                                     .run_config()
        multiplexer = DetectorMultiplexer.create() \
                                         .add_detector(profile_face) \
                                         .add_detector(frontal_face)
        multiplexer.multidetect(cv.imread(argv[1]))
        print(multiplexer.get_regions())
