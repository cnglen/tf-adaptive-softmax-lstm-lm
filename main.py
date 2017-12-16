#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import time

import numpy as np
import tensorflow as tf

from reader import *


def load_config(file="./config/config.json", config_model="DEFAULT"):
    """Load config of neural network from `file` for `config_model`

    Args:
        file (string): path of config file (default "./config/config.json")

    Returns:
        config (dict)

    """
    with open(file, 'r') as f:
        config = json.load(f)

    config[config_model]["adaptive_softmax_cutoff"] = [config[config_model]["adaptive_softmax_cutoff_0"],
                                                       config[config_model]["vocab_size"]]
    config[config_model].pop("adaptive_softmax_cutoff_0", None)
    return config[config_model]


def main():
    config = load_config()
    print(config)


if __name__ == '__main__':
    main()
