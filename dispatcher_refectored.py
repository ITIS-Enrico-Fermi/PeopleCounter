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

    """
