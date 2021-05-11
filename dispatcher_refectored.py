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
        self.source = None  # Either a number (webcam) or a string (video file)
        self.frame: np.ndarray = None
        # self.video_source: str = video_source  # video_source == None if the classifier will be used on an image
        # self.image: str = image  # image == None if the classifier will be used on the video source
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
    def set_preview_processed_frame(self, preview: bool) -> Dispatcher:
        """
        Set processed frame preview
        :return: current object with preview flag set
        """
        self.preview = preview
        return self

    @error_check
    def add_model(self, model_name: str, boundaries: Dict[str, Tuple[int, int]] = None) -> Dispatcher:
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
    

    def __start_time(self) -> None:
        """
        Get current time and save it into self.start_time. Used to compute the elapsed time afterwards
        """
        self.start_time_int = time.time()

    def __end_time(self) -> None:
        """
        Compute elapsed time (between start time and current time) and save it into self.times, in order to figure out what's the average time needed to classify one frame
        """
        logging.info(f"time for 1 frame classification {time.time() - self.start_time_int}")
        if not str.isnumeric(self.video_source):  # If the video source is not a cam
            self.times[self.times_index] = time.time() - self.start_time_int
            self.times_index += 1
    
    def preprocess(self) -> np.ndarray:
        """
        Shared method for frame preprocessing. Frames are preprocessed only once, and then tested against several models, in order to decrease CPU laod and increase recognition speed.
        Final frame (preprocessed) is set as a context variable
        """
        if self.is_first_frame:
            self.display.set_orientation(self.frame)
            self.is_first_frame = False
        self.frame_light = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        self.frame_light = cv.resize(self.frame_light, dsize = self.display.size, interpolation = cv.INTER_AREA)
        # downscaled_frame_gray_equalized: np.ndarray = cv.equalizeHist(downscaled_frame_gray)
        return self.frame_light

    def detect(self) -> List[Region]:
        """
        Detect objects according to the model
        :param np.ndarray frame: frame against which run the classifier
        :param bool processed_frame_preview: am I supposed to show the processed frame?
        :return: a list of regions where the object has been found
        """
        original_frame_regions: List[Region] = list()
        processed_frame_regions: List[Region] = list()
        shape: Shape = Shape.RECTANGLE  # Default shape
        self.__start_time()
        for model, color in zip(self.models_cascade, self.colors):
            processed_frame = self.preprocess(frame)
            obj_list = list()
            if len(set(model_cascade.getOriginalWindowSize())) == 1:  # Face
                shape = Shape.ELLIPSE
                obj_list = model_cascade.detectMultiScale(processed_frame, scaleFactor = 1.2, minSize = config_boundaries['face']['min'], maxSize = config_boundaries['face']['max'])
            else:
                shape = Shape.RECTANGLE
                obj_list = model_cascade.detectMultiScale(processed_frame, scaleFactor = 1.2, minSize = config_boundaries['body']['min'], maxSize = config_boundaries['body']['max'])

            scale_factor_x: float = frame.shape[1] / self.display.size[0]  # both shape[1] and size[0] refer to the x (width)
            scale_factor_y: float = frame.shape[0] / self.display.size[1]  # both shape[0] and size[1] refer to the y (height)
            for (x, y, w, h) in obj_list:
                processed_frame_regions_list.append(Region(x, y, w, h, color, shape))
                original_frame_regions_list.append(Region(x*scale_factor_x, y*scale_factor_y, w*scale_factor_x, h*scale_factor_y, color, shape))
        self.__end_time()
        # if processed_frame_preview:
        #     self.display(processed_frame, processed_frame_regions_list, 'Processed frame preview')
        if not processed_frame_preview:
            return original_frame_regions_list
        else:
            return original_frame_regions_list, processed_frame, processed_frame_regions_list

