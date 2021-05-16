"""
detector_multiplexer_example.py

This script shows how to use DetectorMultiplexer class
"""

__author__ = "Francesco Mecatti & the PeopleCounter Team"

import os
import cv2 as cv
from sys import argv
from detector import DetectorMultiplexer, Detector

class OpencvDetector(Detector):
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
  # The second parameter is the filename
	# of the image to be tested against detectors  
	if len(argv) == 2:
		frontal_face = (OpencvDetector
			.create()
			.set_model('OpencvModels/haarcascade_frontalface_default.xml')
			.run_config())

		profile_face = (OpencvDetector
			.create()
			.set_model('OpencvModels/haarcascade_profileface.xml')
			.run_config())

		multiplexer = (DetectorMultiplexer
			.create()
			.add_detector(profile_face)
			.add_detector(frontal_face))

		multiplexer.multidetect(cv.imread(argv[1]))
		print(multiplexer.get_regions())
