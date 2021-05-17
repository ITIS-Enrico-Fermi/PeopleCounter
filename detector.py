"""
detector.py

Detector abstract class to provide a common interface for different detectors (Haar cascade, Yolo, Darknet, Kersas and so forth)
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import os
import time
from cvlib import *
from abc import ABC, ABCMeta, abstractmethod
from common import context_error

class Detector():
	"""
	Abstract class to provide a
	common interface to different
	detector implementations
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
		self._detector = None
		self._regions: List[Region] = list()  # Detected regions
		self._is_error = False
		self._error: Exception = None
		self._color: Tuple[int, int, int] = None
		self._model = None
		self._shape: Shape = None
		
		self.init()

	@classmethod
	def create(cls):
		return cls()

	@context_error
	def set_model(self, model):
		self._model = model
		return self
	
	@context_error
	def set_color(self, color: Tuple[int, int, int]):
		self._color = color
		return self

	@context_error
	def set_shape(self, shape: Shape):
		self._shape = shape
		return shape

	@context_error
	def bind_detector_implementation(self, detector):
		self._detector = detector
		return self

	def is_error(self) -> bool:
		return self._is_error

	def get_error(self) -> Exception:
		return self._error

	def get_regions(self) -> List[Region]:
		return self._regions

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
		return self

	@abstractmethod
	def config(self) -> None:
		"""
		Initialization function called after setters
		"""
		pass
	
	@abstractmethod
	def detect(self, frame) -> bool:
		"""
		Main method to detect 
		:return: if something has been recognized
		"""
		pass

class DetectorMultiplexer(Detector):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._detectors: List[Detector] = list()
	
	@staticmethod
	def create():
		return DetectorMultiplexer()
	
	def multidetect(self, frame) -> bool:
		"""
		Detect serveral regions with one method
		"""
		self._regions = list()
		for detector in self._detectors:
			detector.detect(frame)
			self._regions.append(detector.get_regions())
		return bool(self._regions)

	@context_error
	def add_detector(self, detector: Detector):
		"""
		Add detector to the context list
		:param Detector detector: detector object
		:return: current instance, with an updated version of _detectors list
		"""
		self._detectors.append(detector)        
		return self


