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
from copy import deepcopy, copy
from collections import defaultdict
from typing import List, Dict

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
		self._roi_history: List[np.ndarray] = list()
		self._blob_history: List[Tuple[int, int, int, int]] = list()
		self._path: List[Tuple[int, int]] = list()
		self._is_init: bool = False

		self.init()
	
	def __copy__(self):
		cls = self.__class__
		rslt = cls.__new__(cls)
		rslt.__dict__.update(self.__dict__)
		return rslt

	def __deepcopy__(self, memo):
		cls = self.__class__
		rslt = cls.__new__(cls)
		memo[id(self)] = rslt
		for k, v in self.__dict__.items():
			if 'Tracker' in v.__class__.__name__ \
				and v.__class__.__module__ == 'cv2':
				setattr(rslt, k, v.__class__.create())
				continue
			setattr(rslt, k, deepcopy(v, memo))
		return rslt	

	@staticmethod
	def register_roi(f):
		def inner(self, frame, *args, **kwargs):
			ok = f(self, frame, *args, **kwargs)
			if ok:
				r = self._region
				cropped = frame[r.y:r.y+r.h, r.x:r.x+r.w]
				self._roi_history.append(cropped)
				
				c = centroid(cropped)
				self._path.append((
					int(c.x+r.x), int(c.y+r.y)
				))
			
			return ok
		return inner

	@staticmethod
	def register_blob(f):
		def inner(self, *args, **kwargs):
			ok = f(self, *args, **kwargs)
			if ok:
				self._blob_history.append( \
					self._region.to_blob())
			return ok
		return inner
	@classmethod
	def create(cls):
		return cls()
	
	def copy(self):
		"""
		Return a deep copy of the current object
		"""
		return deepcopy(self)

	@context_error
	def set_tracker(self, tracker):
		self._tracker = tracker
		return self

	@context_error
	def set_region(self, region):
		"""
		set initialization region
		"""
		self._region = region
		return self
	
	@context_error
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

	def get_roi_history(self) -> List[np.ndarray]:
		return self._roi_history
	
	def get_blob_history(self) -> List[Tuple[int, int, int, int]]:
		return self._blob_history

	def get_path(self) -> List[Tuple[int, int]]:
		return self._path
	
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
		Decorate the implementation with @register_roi
		to keep track of tracked regions over time
		:return: if something has been recognized
		"""
		pass
	
class TrackerMultiplexer(Tracker):
	"""
	Track multiple regionsfo the same frame
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._trackers: List[Tracker] = list()
		self._regions: Dict[int, Region] = defaultdict(Region)
	
	@staticmethod
	def create():
		return TrackerMultiplexer()
	
	def multitrack(self, frame) -> bool:
		self._regions = list()
		for tracker in self._trackers:
			if tracker.track(frame):
				self._regions[id(tracker)] = \
					tracker.get_region()
		return any(self._regions)

	def get_regions(self) -> List[Region]:
		return self._regions

	def get_roi_histories(self) -> Dict[int, List[np.ndarray]]:
		histories = defaultdict(list)
		for tracker in self._trackers:
			histories[id(tracker)] = \
				tracker.get_roi_history()
		return histories

	def get_blob_histories(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
		histories = defaultdict(list)
		for tracker in self._trackers:
			histories[id(tracker)] = \
				tracker.get_blob_history()
		return histories
	
	def get_paths(self) -> Dict[int, List[Tuple[int, int]]]:
		paths = defaultdict(list)
		for tracker in self._trackers:
			paths[id(tracker)] = \
				tracker.get_path()
		return paths


	@context_error
	def add_tracker(self, tracker: Tracker):
		"""
		Add tracker o the context list
		:param Tracker tracker: tracker object
		:return: current instance, with an updated version of _trackers list
		"""
		self._trackers.append(tracker)        
		return self

