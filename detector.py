"""
detector.py

Detector abstract class to provide a common interface for different detectors (Haar cascade, Yolo, Darknet, Kersas and so forth)
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import os
import numpy as np
import time
import cvlib import *

class Detector:
    """
    Abstract class to provide a common interface to different detectors implementation
    """
    
    def __init__(self) -> None:
        """
        Don't override this method. If you want to do some initialization, implement .config() method
        """
        self.__detector = None
        self.__regions: List[Region] = None  # Detected regions
        self.__is_error = False
        self.__error: Error = None
        self.__colors: List[Tuple[int, int, int]] = list()
        self.__model = None

    def error_check(f):
        """
        Decorator for error checking/handling. This way methods don't throw error messages, everything stays inside the context
        """
        def inner(self, *args, **kwargs):
            if self.__is_error:  # Chain of waterfall methods is broken if error
                return self
            try:
                f(self, *args, **kwargs)
            except Error as e:
                self.__is_error = True
                self.__error = e
        return inner

    @error_check
    def set_model(self, model) -> Dispatcher:
        self.__model = model
        return self

    @error_check
    def bind_detector_implementation(self, detector) -> Dispatcher:
        self.__detector = detector
        return self

    def is_error(self) -> bool:
        return self.__is_error

    def get_error(self) -> Error:
        return self.__error

    def get_regions(self) -> List[Region]:
        return self.__regions

    def config(self) -> None:
        """
        Initialization function
        """
        pass

    def detect(self, frame) -> bool:
        """
        Main method to detect 
        :return: if something has been recognized
        """
        pass

class DetectorMultiplexer(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__detectors: List[Detector] = None

    def detect(self, frame: np.ndarray) -> bool:



    @error_check
    def add_detector(self, model_name: str, boundaries: Dict[str, Tuple[int, int]] = None) -> Dispatcher:
        """
        Retrieve path to the model and load
        :param str model_name: relative path to xml model
        :param Dict[str, Tuple[int, int]] boundaries: dict containing min and max size of the object described by the model
        :return: current object, with an updated version of models list
        """
        model = cv.CascadeClassifier()
        model.load(cv.samples.findFile(model_name))
        self.__models.append(model)
        self.__colors = random_colors(len(self.models))
        return self


