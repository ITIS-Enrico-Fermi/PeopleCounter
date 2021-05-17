"""
dispatcher_example.py

Test Dispatcher class
"""

import argparse
import logging
import cv2 as cv
from tracker import Tracker, TrackerMultiplexer
from detector import Detector, DetectorMultiplexer
from dispatcher import Dispatcher
from cvlib import random_colors, Shape
from typing import List, Tuple
from cvlib import Region

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
			# Blobs list
			bl = \
				self._detector.detectMultiScale(frame)
		except:
				return False
		
		# Empty blob list	
		if not len(bl):
			return False
		self._regions = list()
		for b in bl:
			self._regions.append(
				Region(
					*b,
					self._color,
					self._shape))

		return True

class OpencvTracker(Tracker):
	def config(self):
		self._tracker.init(
			self._init_frame,
			self._region.to_blob())
	
	@Tracker.register_blob
	@Tracker.register_roi
	def track(self, frame):
		success, blob = self._tracker.update(frame)
		self._region = \
			Region(*blob,
				self._region.color,
				self._region.shape)
		return success

if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
	
	base_tracker = (OpencvTracker
		.create()
		.set_tracker(
			cv.TrackerKCF
				.create()))

	tracker_multiplexer = (TrackerMultiplexer
			.create())

	colors = random_colors(4)	

	frontal_face = (OpencvDetector
		.create()
		.set_model(
			'OpencvModels/haarcascade_frontalface_default.xml')
		.set_color(colors[0])
		.set_shape(Shape.CIRCLE)
		.run_config())

	profile_face = (OpencvDetector
		.create()
		.set_model(
			'OpencvModels/haarcascade_profileface.xml')
		.set_color(colors[1])
		.set_shape(Shape.CIRCLE)
		.run_config())
	
	full_body = (OpencvDetector
		.create()
		.set_model(
			'OpencvModels/haarcascade_upperbody.xml')
		.set_color(colors[2])
		.set_shape(Shape.RECTANGLE)
		.run_config())

	upper_body = (OpencvDetector
		.create()
		.set_model(
			'OpencvModels/haarcascade_fullbody.xml')
		.set_color(colors[3])
		.set_shape(Shape.RECTANGLE)
		.run_config())

	detector_multiplexer = (DetectorMultiplexer
		.create()
		.add_detector(profile_face)
		.add_detector(frontal_face)
		.add_detector(full_body)
		.add_detector(upper_body))

	dispatcher = (Dispatcher
		.create()
		.bind_tracker(tracker_multiplexer)
		.bind_detector(detector_multiplexer)
		.set_buffer_size(FRAME_BUFFER_SIZE)
		.set_source(0))

	if dispatcher.is_error():
		logging.error(
			"Dispatcher not initialized: "
			f"{dispatcher.get_error()}")
		exit(1)

	tracked_frames = 0
	
	@dispatcher.register_algo
	def dispatching_algo(
			frame,
			dm,	# Detector Multiplexer
			tm,	# Tracker Multiplexer
			d		# Display
		) -> List[Region]:
		
		global tracked_frames
		
		tracked_frames += 1
		rl = list()		

		if tracked_frames >= TRACKING_FRAMES_NUM:
			
			if dm.multidetect(frame):
				logging.debug('detected')
				
				# Regions list
				rl = dm.get_regions()
				
				tm.remove_all()
				
				rl = [r for sr in rl.values() for r in sr]
				for r in rl:
						t = (base_tracker
							.copy()
							.set_frame(frame)
							.set_region(r)
							.run_config())
						tm.add_tracker(t)

				
				tracked_frames = 0
		
		else:
			if tm.multitrack(frame):
				logging.debug('tracked')
				rl = tm.get_regions()
				rl = rl.values()

		d.show(
			frame,
			rl,
			"Haar cascade + KCF")
		
		return rl
	
	dispatcher.loop()
