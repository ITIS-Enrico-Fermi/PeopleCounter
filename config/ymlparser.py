"""
ymlparser.py

YAML utility functions
"""

__author__ = "Emanuele Reggiani & the PeopleCounter team"

import yaml
from typing import Dict

def load_yaml(filename: str) -> Dict:
    with open(filename, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    return config 

def dump_config(config: dict, filename: str) -> None:
    with open(filename, 'w') as file:
        doc = yaml.dump(config, file, default_flow_style=False)

