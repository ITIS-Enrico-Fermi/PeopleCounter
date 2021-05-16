"""
tracker_example.py

This script shows how to use Tracker class
"""

__author__ = "Francesco Mecatti & the PeopleCounter Team"

import os
import cv2 as cv
from tracker import Tracker
from sys import argv
from cvlib import Shape, Region, Display

class PeopleTracker(Tracker):
	def config(self):
		self._tracker.init(
			self._init_frame,
			self._region.to_blob()
		)
	
	def track(self, frame):
		success, blob = self._tracker.update(frame)
		self._region = \
			Region(*blob,
				self._region.color,
				self._region.shape)
		return success

if __name__ == "__main__":
	# The second argument is
	# the filename of a video
	if len(argv) == 2:
		tracker_implementation = \
			cv.TrackerKCF_create()	
	
		t = (PeopleTracker
			.create()
			.set_tracker(
				tracker_implementation))

		d = Display()
		
		cap = cv.VideoCapture(argv[1])
		while True:
			ret, frame = cap.read()
			if frame is None:
				break
			if cv.waitKey(1) == 27:
				break
			if not t.is_init():
				# The user selects one or
				# more regions of interest
				rois = cv.selectROI(frame)
				(t
					.set_frame(frame)
					.set_region(Region(*rois, (255, 0, 0), Shape.RECTANGLE))
					.run_config())
			
				if t.is_error():
					print(f"An error occurred: {t.get_error()}")
				
				continue

			t.track(frame)
			d.show(
				frame,
				[t.get_region()],
				"Tracking"
			)
				
