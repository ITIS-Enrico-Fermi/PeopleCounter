"""
tracker.py

Tracker abstract class to provide a common interface for different trackers 
A Tracker objects keeps paths, shots ecc ecc inside its context
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import os
import time
from cvlib import *
from abc import ABC, ABCMeta, abstractmethod
from common import context_error

class Tracker():
	"""
	Abstract class to provide a
	common interface to different
	tracker implementations
	Create a sublcass in which
	detect() and config() are implemented
	"""

	__metaclass__ = ABCMeta
	
	def __init__(self):
		"""
		Don't override this method
		If you want to do some initialization before
		setter methods are called, implement .init()
		"""
		self._tracker = None
		self._region: Region = None  # Init region, after that tracked region
		self._init_frame = None
		self._is_error = False
		self._error: Exception = None
		self._blob_history: List[Tuple[int, int, int, int]] = None
		self._is_init: bool = False

		self.init()

	@classmethod
	def create(cls):
		return cls()

	@context_error
	def set_tracker(self, tracker):
		self._tracker = tracker
		return self

	def set_region(self, region):
		"""
		set initialization region
		"""
		self._region = region
		return self

	def set_frame(self, frame):
		"""
		set initialization frame
		"""
		self._init_frame = frame
		return self

	def is_error(self) -> bool:
		return self._is_error

	def get_error(self) -> Exception:
		return self._error

	def get_region(self) -> Region:
		return self._region

	def get_history(self) -> List[Tuple[int, int, int, int]]:
		return self._blob_history

	def is_init(self) -> bool:
		return self._is_init

	@abstractmethod
	def init(self) -> None:
		"""
		Initialization function called before setter methods
		"""
		pass
	
	@context_error
	def run_config(self):
		"""
		Run config in a "safe" way, keeping errors inside the context
		This should be the last call of the chained methods stack
		"""
		self.config()
		# Init process completed
		self._is_init = True
		return self

	@abstractmethod
	def config(self) -> None:
		"""
		Initialization function called after setters
		Initialize your tracker here
		"""
		pass

	@abstractmethod
	def track(self, frame) -> bool:
		"""
		Main method to track objects
		:return: if something has been recognized
		"""
		pass
'''
class DetectorMultiplexer(Detector):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._detectors: List[Detector] = list()
	
	@staticmethod
	def create():
		return DetectorMultiplexer()
	
	def multidetect(self, frame) -> bool:
		for detector in self._detectors:
			detector.detect(frame)
			self._regions.append(detector.get_regions())

	@context_error
	def add_detector(self, detector):
		"""
		Add detector to the context list
		:param Detector detector: detector object
		:return: current instance, with an updated version of __detectors list
		"""
		self._detectors.append(detector)        
		return self

'''
