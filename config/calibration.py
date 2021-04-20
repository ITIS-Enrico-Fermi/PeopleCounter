import cv2 as cv
import argparse
import logging
import os
import numpy as np
from typing import Tuple, List
import yaml


def load_yaml(filename: str) -> dict:
	with open(filename, 'r') as file:
		config = yaml.load(file, Loader=yaml.FullLoader)
	return config 

def dump_config(config: dict, filename: str) -> None:
	with open(filename, 'w') as file
		doc = yaml.dump(config, file)
