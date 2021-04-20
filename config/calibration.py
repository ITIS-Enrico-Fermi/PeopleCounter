import cv2 as cv
import argparse
import logging
import os
import numpy as np
from typing import Tuple, List
import yaml


def load_yaml(filename: str) -> dict:
	with open(filename) as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
	return config 