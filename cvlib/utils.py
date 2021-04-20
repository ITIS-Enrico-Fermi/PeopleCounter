import cv2 as cv
import argparse
import logging
import os
import numpy as np
import time
from typing import Tuple, List
from enum import Enum, auto
from math import floor, ceil

class Point:
    """
    Class representing a point (x, y)
    """
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x: int = x
        self.y: int = y

    def to_tuple(self) -> Tuple[int, int]:
        """
        Useful method to turn the coordinates into a tuple
        :return: the point as a tuple (x, y)
        """
        return (self.x, self.y)

class Shape(Enum):
    """
    Enum to provide shape to cv methods mapping
    """
    RECTANGLE: int = auto()
    CIRCLE: int = auto()
    ELLIPSE: int = auto()

class Region:
    """
    Class representing a ROI - Region Of Interest - (x, y, w, h)
    """
    def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0, color: Tuple[int, int, int] = (0, 255, 0), shape: Shape = Shape.RECTANGLE) -> None:
        """
        Object constructor
        x, y: coordinates of upper left point
        w, h: size of the region
        """
        self.x: int = int(x)
        self.y: int = int(y)
        self.w: int = int(w)
        self.h: int = int(h)
        self.color: Tuple[int, int, int] = color
        self.shape: Shape = shape

    def get_area(self) -> int:
        """
        Get ROI's area
        :return: the area
        """
        return (self.w * self.h)

    def get_center(self) -> Point:
        """
        Get ROI center
        :return: coordinates of ROI's cental point
        """
        return Point(self.x + self.w // 2, self.y + self.h // 2)
    
    def get_upper_left(self) -> Point:
        """
        Returns coordinate of the upper-left corner
        """
        return Point(self.x, self.y)

    def get_bottom_right(self) -> Point:
        """
        Returns coordinate of the bottom-right corner
        """
        return Point(self.x + self.w, self.y + self.h)

class Orientation(Enum):
    """
    Enum representing the orientation of a frame/image
    """
    VERTICAL: int = auto()
    HORIZONTAL: int = auto()
    SQUARE: int = auto()

    @staticmethod
    def get_orientation(img: np.ndarray):
        """
        Get img's orientation
        :param np.ndarray img: image you want to get the orientation of
        :return: orientation of the image
        """
        w: int = img.shape[1]
        h: int = img.shape[0]
        if w > h:
            return Orientation.HORIZONTAL
        elif h > w:
            return Orientation.VERTICAL
        else:
            return Orientation.SQUARE

class Color:
    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r
        self.g = g
        self.b = b

def random_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    :param int n: number of color sets
    Return a list of random colors equally spaced
    """
    colors: List[Tuple[int, int, int]] = list()
    equally_spaced_colors: np.array = np.linspace(0, 256, num=n*3, dtype=int)
    np.random.shuffle(equally_spaced_colors)
    r_array: np.array = equally_spaced_colors
    np.random.shuffle(equally_spaced_colors)
    g_array: np.array = equally_spaced_colors
    np.random.shuffle(equally_spaced_colors)
    b_array: np.array = np.random.choice(range(256), size=n)
    
    return list(zip(r_array.tolist(), g_array.tolist(), b_array.tolist()))

def scale(img: np.ndarray, scale_factor: float, min_size: Tuple[int, int] = (0, 0)) -> np.ndarray:  # scale_factor between 0 and 1 if you want to scale down the image
    """
    Scale an image with a scale factor
    :param np.ndarray image: original image
    :param fload scale_factor: between 1 and 0 if you want to downscale the image. Scale factor bigger than 1 will increse the size of the image
    """
    h = img.shape[0]
    w = img.shape[1]
    scaled_h: int = int(h * scale_factor)
    scaled_w: int = int(w * scale_factor)
    if max(scaled_h, scaled_w) < min(min_size):
        if w > min_size[0] and w - min_size[0] == max(w - min_size[0], h - min_size[1]):
            scaled_w = min_size[0]
            scaled_h = (h/w)*min_size[0]
        else:
            scaled_h = min_size[1]
            scaled_w = (w/h)*min_size[1]
    return cv.resize(img, (int(scaled_w), int(scaled_h)))