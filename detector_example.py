"""
detector_example.py

This script shows how to use Detector class
"""

__author__ = "Francesco Mecatti & the PeopleCounter Team"

import cv2 as cv
from detector import Detector
import os
from sys import argv

class FrontalFaceDetector(Detector):
	def config(self):
		filename = self._model
		detector = cv.CascadeClassifier()
		detector.load(filename)
		self._detector = detector

	def detect(self, frame):
		try:
			self._regions = \
				self._detector.detectMultiScale(frame)
		except:
			return False		
		return True

if __name__ == "__main__":
	if len(argv) == 2:  # The second parameter is the filename of the image to be tested against detectors
		d = (FrontalFaceDetector
			.create()
			.set_model('OpencvModels/haarcascade_frontalface_default.xml')
			.run_config())

		d.detect(cv.imread(argv[1]))
		print(d.get_regions())
