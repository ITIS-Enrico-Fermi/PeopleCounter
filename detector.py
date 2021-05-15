"""
detector.py

Detector abstract class to provide a common interface for different detectors (Haar cascade, Yolo, Darknet, Kersas and so forth)
"""

__author__ = "Francesco Mecatti & the PeopleCounter team"

import cv2 as cv
import os
import numpy as np
import time
from cvlib import *

def context_error(f):
    """
    Decorator for error checking/handling. This way methods don't throw error messages, everything stays inside the context
    """
    def inner(*args, **kwargs):
        self = args[0]
        # Chain of waterfall methods is broken if error
        try:
            f(*args, **kwargs)
        except Exception as e:
            self._is_error = True
            self._error = e
        finally:
            return self
    return inner

class Detector:
    """
    Abstract class to provide a common interface to different detectors implementation
    """
    
    def __init__(self):
        """
        Don't override this method. If you want to do some initialization, implement .config() method
        """
        self._detector = None
        self._regions: List[Region] = None  # Detected regions
        self._is_error = False
        self._error: Exception = None
        self._colors: List[Tuple[int, int, int]] = list()
        self._model = None
    
    @staticmethod
    def create():
        return Detector()

    @context_error
    def set_model(self, model):
        self._model = model
        return self

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

    def config(self) -> None:
        """
        Initialization function
        """
        pass

    def detect(self, frame: np.ndarray) -> bool:
        """
        Main method to detect 
        :return: if something has been recognized
        """
        pass

class DetectorMultiplexer(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detectors: List[Detector] = None

    def multidetect(self, frame: np.ndarray) -> bool:
        for detector in self.__detectors:
            self._regions.append(detector.detect(frame))
    
    @context_error
    def add_detector(self, detector):
        """
        Add detector to the context list
        :param Detector detector: detector object
        :return: current instance, with an updated version of __detectors list
        """
        self._detectors.append(detector)        
        return self


