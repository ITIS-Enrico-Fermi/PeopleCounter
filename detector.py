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

def error_check(f):
    """
    Decorator for error checking/handling. This way methods don't throw error messages, everything stays inside the context
    """
    print(f)
    def inner(*args, **kwargs):
        self = args[0]
        print(args)
        # Chain of waterfall methods is broken if error
        try:
            f(*args, **kwargs)
        except Exception as e:
            print('error')
            print(self.__is_error)
            self.__is_error = True
            print(self.__is_error)
            self.__error = e
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
        self.__detector = None
        self.__regions: List[Region] = None  # Detected regions
        self.__is_error = False
        self.__error: Exception = None
        self.__colors: List[Tuple[int, int, int]] = list()
        self.__model = None
    
    @staticmethod
    def create():
        return Detector()

    @error_check
    def set_model(self, model):
        raise Exception('test')
        self.__model = model
        return self

    @error_check
    def bind_detector_implementation(self, detector):
        self.__detector = detector
        return self

    def is_error(self) -> bool:
        return self.__is_error

    def get_error(self) -> Exception:
        return self.__error

    def get_regions(self) -> List[Region]:
        return self.__regions

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
        self.__detectors: List[Detector] = None

    def multidetect(self, frame: np.ndarray) -> bool:
        for detector in self.__detectors:
            self.__regions.append(detector.detect(frame))
    
    @error_check
    def add_detector(self, detector):
        """
        Add detector to the context list
        :param Detector detector: detector object
        :return: current instance, with an updated version of __detectors list
        """
        self.__detectors.append(detector)        
        return self


