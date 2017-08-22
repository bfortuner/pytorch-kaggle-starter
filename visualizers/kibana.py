import json
import pandas as pd
import config
import constants as c
import clients.client_constants as cc
import clients.es_client as es
import copy


class Kibana():
    
    def __init__(self, exp_name):
        self.name = exp_name
        self.classname = 'Kibana'

    def init(self, exp_config):
        assert config.ES_ENABLED is True
        assert es.ping() is True

    def update(self, exp_config, exp_history, msg=None):
        es.upload_experiment_history(exp_config, exp_history)
        es.upload_experiment_config(exp_config)


def load(config):
    return Kibana(config.name)


