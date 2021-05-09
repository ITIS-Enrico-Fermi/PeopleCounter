import cv2 as cv
import argparse
import logging
import os
import numpy as np
import time
from typing import Tuple, List
from enum import Enum, auto
from math import floor, ceil
from .utils import *

VGA_HORIZONTAL_SIZE: Tuple[int, int] = (640, 480)
VGA_VERTICAL_SIZE: Tuple[int, int] = tuple(reversed(VGA_HORIZONTAL_SIZE))

class Display:
    def __init__(self):
        self.size: Tuple(int, int) = (0, 0)  # This param will contain the destionation size of each frame. Filled after the first frame is processed
        self.orientation: Orientation = None  # Set after the first frame is sampled

    def show(self, frame: np.ndarray, regions: List[Region], window_title: str = 'OpenCV show image', scale_factor: float = 1.0, frame_processed: np.ndarray = None, regions_processed: List[Region] = None) -> None:
        """
        Display a frame drawing a series of ellipses around the regions of interest
        :param np.ndarray frame: original frame
        :param List[Region] regions: regions of interest list
        :param str window_title: window's title
        :param float scale_factor: the frame will be scaled according to this value for better view
        """
        if regions:
            for region in regions:
                frame: np.ndarray = draw(frame, region)
        frame = scale(frame, scale_factor, self.size)
        if frame_processed is not None and regions_processed is not None:
            for region_processed in regions_processed:
                frame_processed = draw(frame_processed, region_processed)
            fh, fw = frame.shape[:2]
            fph, fpw = frame_processed.shape[:2]
            frame_processed = cv.copyMakeBorder(frame_processed, floor((fh-fph)/2) if fh>fph else 0, ceil((fh-fph)/2) if fh>fph else 0, floor((fw-fpw)/2) if fw>fpw else 0, ceil((fw-fpw)/2) if fw>fpw else 0, cv.BORDER_CONSTANT)
            frame = cv.copyMakeBorder(frame, floor((fph-fh)/2) if fph>fh else 0, ceil((fph-fh)/2) if fph>fh else 0, floor((fpw-fw)/2) if fpw>fw else 0, ceil((fpw-fw)/2) if fpw>fw else 0, cv.BORDER_CONSTANT)
            frame = np.concatenate((frame, cv.cvtColor(frame_processed, cv.COLOR_GRAY2BGR)), axis=0 if self.orientation is Orientation.HORIZONTAL else 1)
        # cv.putText(frame, "test", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv.imshow(window_title, frame)
        # if not self.main_window_created:
        #     cv.moveWindow(window_title, 100, 100)
        #     self.main_window_created = True

    def set_orientation(self, frame: np.ndarray):
            self.orientation = Orientation.get_orientation(frame)
            if self.orientation is Orientation.VERTICAL:
                self.size: Tuple[int, int] = VGA_VERTICAL_SIZE
            elif self.orientation is Orientation.HORIZONTAL:
                self.size: Tuple[int, int] = VGA_HORIZONTAL_SIZE

def draw_circle(frame: np.ndarray, region: Region) -> np.ndarray:
    """
    Draw a circle around a ROI
    :param np.ndarray frame: original frame
    :pram Region region: ROI around which drawing the circle
    :return: new frame on with the circle
    """
    return cv.circle(frame, region.get_center().to_tuple(), region.w // 2, region.color, 4)

def draw_rectangle(frame: np.ndarray, region: Region) -> np.ndarray:
    """
    Draw rectangle around a ROI
    :param np.ndarray frame: original frame
    :pram Region region: ROI around which drawing the rectangle
    :return: new frame on with the rectangle
    """
    return cv.rectangle(frame, region.get_upper_left().to_tuple(), region.get_bottom_right().to_tuple(), region.color, 4)

def draw_ellipse(frame: np.ndarray, region: Region) -> np.ndarray:
    """
    Draw ellipse around a ROI
    :param np.ndarray frame: original frame
    :pram Region region: ROI around which drawing the ellipse
    :return: new frame on with the ellipse
    """
    return cv.ellipse(frame, region.get_center().to_tuple(), (region.w // 2, region.h // 2), 0, 0, 360, region.color, 4)

def draw(frame: np.ndarray, region: Region) -> np.ndarray:
    if region.shape is Shape.RECTANGLE:
        return draw_rectangle(frame, region)
    elif region.shape is Shape.ELLIPSE:
        return draw_ellipse(frame, region)
    elif region.shape is Shape.CIRCLE:
        return draw_circle(frame, region)
    else:
        return frame


