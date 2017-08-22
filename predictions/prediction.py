import os
import copy

import config as cfg
import constants as c
from clients import s3_client
from clients import es_client
from .pred_constants import *



class Prediction:
    def __init__(self, fpath, metadata):
        self.fpath = fpath
        self.meta = metadata

    @property
    def name(self):
        return os.path.basename(self.fpath).rstrip(
            c.PRED_FILE_EXT)

    @property
    def id(self):
        return self.name.split('-id')[-1]

    @property
    def display_name(self):
        return self.name.split('-id')[0]

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_doc(self):
        dict_ = self.to_dict()
        dict_['key'] = self.id
        dict_['display_name'] = self.display_name()
        return dict_

    def save(self, s3=cfg.S3_ENABLED, es=cfg.ES_ENABLED):
        if s3: 
            s3_client.upload_prediction(self.fpath, self.name())
        if es: 
            es_client.upload_prediction(self)
