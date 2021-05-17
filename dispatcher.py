"""
dispatcher.py

Dispatcher acts as the central component of the architecture, serving frames to the tracker or the detector according to the provided algorithm.
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import argparse
import logging
import os
import numpy as np
import time
from typing import Tuple, List
from enum import Enum, auto
from math import floor, ceil
from cvlib import *
from config import config_boundaries

class Dispatcher:
	"""
	Core of the entire architecture
	Remember to add tracker and classifier object
	The user must implement dispatching algorithm
	No errors will be thrown, if an error occurs the state of 'error' context variable will turn to True
	"""

	def __init__(self) -> None:
		"""
		Constructor of the class Classifier
		:param List[str] models: relative path to the xml models
		:param str video_source: video source. If video_source is a string, it's supposed to be the relative path to a file, else video_source is converted to an integer and the video stream is treated like a cam
		"""
		self._models: List[cv.CascadeClassifier] = list()
		self._source = None  # Either a number (webcam) or a string (video file)
		self._frame: np.ndarray = None
		# self.video_source: str = video_source  # video_source == None if the classifier will be used on an image
		# self.image: str = image  # image == None if the classifier will be used on the video source
		self._start_time_int: int = None  # start_time will fill this attribute for the first time
		self._times: np.array = None  # start will fill this attribute
		self._times_index: int = 0  # Index to keep track of times array filling
		# self.main_window_created: bool = False
		self._is_first_frame: bool = True
		self._colors: List[Tuple[int, int, int]] = None
		self._display: Display = Display()
		self._tracked_frames: int = 0
		self._tracker: cv.Tracker = None
		self._is_tracker_init: bool = False
		self._is_error: bool = False
		self._error: Error = None

	@staticmethod
	def create():
		"""
		Create and return Dispatcher object
		:return: new object of this class
		"""
		return Dispatcher()

	@error_check
	def set_source(self, source):
		"""
		Set video source
		:param str source: video source. It can be either a webcam (integer) or a file (string)
		:return: current object with source set
		"""
		self._source = int(source) if str.isnumeric(source) else source
		return self

	@error_check
	def set_preview_processed_frame(self, preview: bool):
		"""
		Set processed frame preview
		:param bool preview: True when the preprocessed frame should be displayed
		:return: current object with preview flag set
		"""
		self._preview = preview
		return self

	@context_error
	def bind_tracker(self, tracker: Tracker):
		"""
		Bind tracker object. The tracker must implement .init() and .update() methods. See Tracker interface
		:param Tracker tracker: implementation of Tracker interface
		:return: Current object, with tracker set
		"""
		# TODO check class and set error, implement Tracker
		self._tracker = tracker
		return self

	@error_check
	def bind_detector(self, detector: Detector):
		"""
		Bind detector object. See Detector interface
		:param Detector detector: implementation of Detector interface
		:return: Current object with detector binded
		"""
		# TODO check class and set error, implement Detector
		self._detector = detector
		return self
	
	def register_algo(self, f):
		"""
		Decorator for dispatching algo registration
		"""
		self._dispatching_algo = f
		return f

	def __start_time(self) -> None:
		"""
		Get current time and save it into self.start_time. Used to compute the elapsed time afterwards
		"""
		self._start_time_int = time.time()

	def __end_time(self) -> None:
		"""
		Compute elapsed time (between start time and current time) and save it into self.times, in order to figure out what's the average time needed to classify one frame
		"""
		logging.info(f"time for 1 frame classification {time.time() - self.start_time_int}")
		if not str.isnumeric(self.video_source):  # If the video source is not a cam
			self._times[self._times_index] = time.time() - self._start_time_int
			self._times_index += 1

	def preprocess(self):
		"""
		Shared method for frame preprocessing. Frames are preprocessed only once, and then tested against several models, in order to decrease CPU laod and increase recognition speed.
		Final frame (preprocessed) is set as a context variable
		"""
		if self._is_first_frame:
			self.display.set_orientation(self.frame)
			self.is_first_frame = False
		self.frame_light = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
		self.frame_light = cv.resize(self.frame_light, dsize = self.display.size, interpolation = cv.INTER_AREA)
		# downscaled_frame_gray_equalized: np.ndarray = cv.equalizeHist(downscaled_frame_gray)
		# return self.frame_light  # New frame is both set as a context variable and returned from by this method
		return self
	
    def loop(self) -> None:  # Blocking method
			"""
			Start video capture and frames classification. Be aware that it's a blocking method (it enters a loop)
			:param bool processed_frame_preview: am I supposed to show the processed frame?
			"""
			if self.image:
				img: np.ndarray = cv.imread(self.image)
				self.detect_and_display(img, processed_frame_preview)
				if cv.waitKey(0) == 27:  # Key ==> 'ESC'
					return

			cap = cv.VideoCapture(int(self.video_source) if str.isnumeric(self.video_source) else self.video_source)
			frames_number: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
			if frames_number > 0:  # frames_num < 0 when the video source is a camera
				self.times = np.empty(frames_number, dtype='f', order='C')
				cap.set(cv.CAP_PROP_BUFFERSIZE, FRAME_BUFFER_SIZE)
			if not cap.isOpened():
				logging.error("Camera video stream can't be opened")
				exit(1)
			while True:
				ret, frame = cap.read()
				if frame is None:
					break
				self.dispatch(frame, processed_frame_preview)
				# self.detect_and_display(frame, processed_frame_preview)
				if cv.waitKey(1) == 27:  # Key ==> 'ESC'
					break
			# When classification is done, print the average time needed to classify each frame
			if frames_number > 0:
				logging.info(f"Average time needed to classify each frame {np.average(self.times[:self.times_index])}")
				logging.info(f"Max time needed to classify each frame {np.amax(self.times[:self.times_index])}")
				logging.info(f"Min time needed to classify each frame {np.amin(self.times[:self.times_index])}")
