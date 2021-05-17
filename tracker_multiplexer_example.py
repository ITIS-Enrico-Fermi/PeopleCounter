"""
tracker_multiplexer_example.py

This script shows how to use Tracker class
"""

__author__ = "Francesco Mecatti & the PeopleCounter Team"

import os
import cv2 as cv
from sys import argv
import numpy as np
from cvlib import Shape, Region, Display
from tracker import Tracker, TrackerMultiplexer 

class PeopleTracker(Tracker):
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
	# The second argument is
	# the filename of a video
	if len(argv) == 2:
		tracker_implementation = \
			cv.TrackerKCF_create()	
	
		t = (PeopleTracker
			.create()
			.set_tracker(
				tracker_implementation))

		tm = (TrackerMultiplexer
			.create())

		d = Display()
		
		cap = cv.VideoCapture(argv[1])
		last_frame = None
		while True:
			ret, frame = cap.read()
			if frame is None:
				break
			last_frame = frame
			if cv.waitKey(1) & 0xFF == 27:
				break
			if not t.is_init():
				# The user selects one or
				# more regions of interest
				while True:
					roi =	cv.selectROI(frame)
					t = (t
						.copy()
						.set_frame(frame)
						.set_region(
							Region(
								*roi,
								(255, 0, 0),
								Shape.RECTANGLE))
						.run_config())
					tm.add_tracker(t)
					
					# Press ENTER for the second time
					# (the first one confirms roi selection)
					# to add another region to the list
					print('Press ENTER to select another region')
					if cv.waitKey(0) & 0xFF != 13:  # Wait forever
						break
			
				if t.is_error():
					print(f"An error occurred: {t.get_error()}")
				
				continue
			
			tm.multitrack(frame)
			d.show(
				frame,
				tm.get_regions(),
				"Tracking"
			)
		
		# Show paths of the last obj tracked
		path = tm.get_paths()[id(t)]
		frame = cv.polylines(
			last_frame,
			[np.array(path)],
			isClosed = False,
			color = (255, 0, 0),
			thickness = 3)
		print(frame)
		d.show(
			frame,
			[],
			"Path"
		)

		if cv.waitKey(0) & 0xFF == 27:
			pass

		# print(tm.get_blob_histories())
		# Show only history of the last tracker
		for roi in tm.get_roi_histories()[id(t)]:
			d.show(
				roi,
				[],
				"Blob history"
			)
			if cv.waitKey(0) & 0xFF == 27:
				break
