import os
import numpy as np
import pandas as pd

import config as cfg
import constants as c

import datasets.metadata as meta
import utils


LABEL_NAMES = ['cat', 'dog']
LABEL_TO_IDX = meta.get_labels_to_idxs(LABEL_NAMES)
IDX_TO_LABEL = meta.get_idxs_to_labels(LABEL_NAMES)
SUB_HEADER = 'id,label'