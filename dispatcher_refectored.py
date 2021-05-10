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
from config import config_boundarys

FRAME_BUFFER_SIZE = 1
TRACKING_ALGO = 'KCF'
TRACKING_FRAMES_NUM = 100

class Dispatcher:
    """
    Core of the entire architecture
    Remember to add tracker and classifier object
    The user must implement dispatching algorithm
    No errors will be thrown, if an error occurs the state of 'error' context variable will turn to True
    """

    def __init__(self, video_source: str = None, image = None) -> None:
        """
        Constructor of the class Classifier
        :param List[str] models: relative path to the xml models
        :param str video_source: video source. If video_source is a string, it's supposed to be the relative path to a file, else video_source is converted to an integer and the video stream is treated like a cam
        """
        self.models_cascade: List[cv.CascadeClassifier] = list()
        self.video_source: str = video_source  # video_source == None if the classifier will be used on an image
        self.image: str = image  # image == None if the classifier will be used on the video source
        self.start_time_int: int = None  # start_time will fill this attribute for the first time
        self.times: np.array = None  # start will fill this attribute
        self.times_index: int = 0  # Index to keep track of times array filling
        # self.main_window_created: bool = False
        self.is_first_frame: bool = True
        self.colors: List[Tuple[int, int, int]] = random_colors(len(models))
        self.display: Display = Display()
        self.tracked_frames: int = 0
        self.prev_frame: np.ndarray = None
        self.prev_regions: List[Region] = None
        self.tracker: cv.Tracker = None
        self.is_tracker_init: bool = False
        self.is_error: bool = False
        self.error: Error = None

    @staticmethod
    def create() -> Dispatcher:
        """
        Create and return Dispatcher object
        :return: new object of this class
        """
        return Dispatcher()
    
    def error_check(f):
        """
        Decorator for error checking/handling. This way methods don't throw error messages, everything stays inside the context
        """
        def inner(self, *args, **kwargs):
            if self.error:  # Chain of waterfall methods is broken if error
                return self
            try:
                f(self, *args, **kwargs)
            except Error as e:
                self.is_error = True
                self.error = e
        return inner

    @error_check
    def set_source(self, source: str) -> Dispatcher:
        """
        Set video source
        :return: current object with source set
        """
        self.source = int(source) if str.isnumeric(source) else source
        return self

    @error_check
    def add_model(self, model_name: str) -> Dispatcher:
        """
        Retrieve path to the model and load it
        :return: current object, with an updated version of models list
        """
        model = cv.CascadeClassifier()
        model.load(cv.samples.findFile(model_name))
        self.models_cascade.append(model)
        return self
    
    @error_check
    def bind_tracker(self, tracker: Tracker) -> Dispatcher:
        """
        Bind tracker object. The tracker must implement .init() and .update() methods. See Tracker interface
        :return: Current object, with tracker set
        """
        # TODO check class and set error, implement Tracker
        self.tracker = tracker
        return self

    @error_check
    def bind_detector(self, detector: Detector) -> Dispatcher:
        """
        Bind detector object. See Detector interface
        :return: Current object with detector binded
        """
        # TODO check class and set error, implement Detector
        self.detector = detector
        return self

    def register_algo(f):
        """
        Decorator for dispatching algo registration
        """
        def inner(self):
            self.dispatching_algo = f
        return f
