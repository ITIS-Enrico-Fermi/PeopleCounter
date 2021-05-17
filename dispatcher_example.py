"""
dispatcher_example.py

Test Dispatcher class
"""

import argparse
import logging
from tracker import Tracker, TrackerMultiplexer
from detector import Detector, DetectorMultiplexer
from dispatcher import Dispatcher

FRAME_BUFFER_SIZE = 1
TRACKING_FRAMES_NUM = 100

dispatcher = None

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

class OpencvTracker(Tracker):
	def config(self):
		self._tracker.init(
			self._init_frame,
			self._region.to_blob()
		)
	
	@Tracker.register_blob
	@Tracker.register_roi
	def track(self, frame):
		success, blob = self._tracker.update(frame)
		self._region = \
			Region(
				*blob,
				self._region.color,
				self._region.shape
			)
		return success

if __name__ == "__main__":
	base_tracker = (OpencvTracker
		.create()
		.set_tracker(
			cv.TrackerKCF.create))

	tracker_multiplexer = (TrackerMultiplexer
			.create())

	dispatcher = \
		Dispatcher
		.create()
		.bind_tracker()
		.bind_detector()
	
	@dispatcher.register_algo
	def dispatching_algo(
		frame,
		detector_multiplexer,
		tracker_multiplexer)
		-> None:
	
		
