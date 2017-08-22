import os
import json
import pprint
import logging
import time
import copy
from datetime import datetime

import pandas as pd

import config
import constants as c
import utils.files
import utils.general
from clients import s3_client, es_client




class ExperimentConfig():
    def __init__(self, name, parent_dir, created, metrics, aux_metrics,
                model, optimizer, lr_adjuster, criterion, transforms,
                visualizers, training, data, hardware, other, progress=None):
        self.name = name
        self.parent_dir = parent_dir
        self.fpath = os.path.join(parent_dir, name,
                                  c.EXPERIMENT_CONFIG_FNAME)
        self.created = created
        self.metrics = metrics
        self.aux_metrics = aux_metrics
        self.visualizers = visualizers
        self.model = model
        self.optimizer = optimizer
        self.lr_adjuster = lr_adjuster
        self.criterion = criterion
        self.transforms = transforms
        self.data = data
        self.training = training
        self.hardware = hardware
        self.other = other
        self.progress = {} if progress is None else progress
        self.model_name = self.model['name']

    def get_id(self):
        return self.name.split('-id')[-1]

    def get_display_name(self):
        return self.name.split('-id')[0]

    def summary(self, include_model=True):
        d = dict(self.__dict__)
        del d['model']
        print(json.dumps(d, indent=4, ensure_ascii=False))
        if include_model:
            print(self.model['layers'])

    def save(self, s3=config.S3_ENABLED, es=config.ES_ENABLED):
        dict_ = self.__dict__
        utils.files.save_json(self.fpath, dict_)
        if s3:
            s3_client.upload_experiment_config(self.fpath, self.name)
        if es:
            es_client.upload_experiment_config(self)

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    def to_html(self):
        dict_ = self.to_dict()
        html = utils.general.dict_to_html_ul(dict_)
        return html

    def to_doc(self):
        # Changes to self.__dict__ also change instance variables??
        doc = copy.deepcopy(self.to_dict())
        doc[c.EXP_ID_FIELD] = self.get_id()
        doc[c.ES_EXP_KEY_FIELD] = self.get_id()
        doc['display_name'] = self.get_display_name()
        doc['transforms'] = str(doc['transforms'])
        del doc['model']
        return doc


## Helpers

def fetch_external_config(exp_name):
    str_ = s3_client.fetch_experiment_config(exp_name)
    dict_ = json.loads(str_)
    return load_config_from_json(dict_)


def load_config_from_file(fpath):
    dict_ = utils.files.load_json(fpath)
    return load_config_from_json(dict_)


def load_config_from_json(dict_):
    return ExperimentConfig(
            name=dict_['name'],
            parent_dir=dict_['parent_dir'],
            created=dict_['created'],
            metrics=dict_['metrics'],
            aux_metrics=dict_['aux_metrics'],
            visualizers=dict_['visualizers'],
            model=dict_['model'],
            optimizer=dict_['optimizer'],
            lr_adjuster=dict_['lr_adjuster'],
            criterion=dict_['criterion'],
            transforms=dict_['transforms'],
            data=dict_['data'],
            training=dict_['training'],
            hardware=dict_['hardware'],
            other=dict_['other'],
            progress=dict_['progress'])


def create_config_from_dict(config):
    metrics_config = get_metrics_config(config['metrics'])
    aux_metrics_config = get_aux_metrics_config(config['aux_metrics'])
    visualizers_config = get_visualizers_config(config['visualizers'])
    transforms_config = get_transforms_config(config['transforms'])
    model_config = get_model_config(config['model'])
    optim_config = get_optim_config(config['optimizer'])
    lr_adjuster_config = get_lr_config(config['lr_adjuster'])
    criterion_config = get_criterion_config(config['criterion'])
    return ExperimentConfig(
            name=config['name'],
            parent_dir=config['parent_dir'],
            created=time.strftime("%m/%d/%Y %H:%M:%S", time.localtime()),
            metrics=metrics_config,
            aux_metrics=aux_metrics_config,
            visualizers=visualizers_config,
            model=model_config,
            optimizer=optim_config,
            lr_adjuster=lr_adjuster_config,
            criterion=criterion_config,
            transforms=transforms_config,
            data=config['data'],
            training=get_training_config(config['training']),
            hardware=config['hardware'],
            other=config['other'])


def remove_large_items(dict_):
    max_len = 100
    new_dict = {}
    for k,v in dict_.items():
        if isinstance(v, list) and len(v) > max_len:
            pass
        elif isinstance(v, dict):
            if len(v.items()) < max_len:
                new_dict[k] = str(v.items())
        else:
            assert not isinstance(v, dict)
            new_dict[k] = v
    return new_dict


def get_training_config(train_config):
    return remove_large_items(train_config)


def get_model_config(model):
    name = utils.general.get_class_name(model)
    layers = str(model)
    return {
        'name': name,
        'layers': layers
    }


def get_optim_config(optim):
    name = utils.general.get_class_name(optim)
    params = optim.param_groups[0]
    params = remove_large_items(dict(params))
    if 'params' in params:
        del params['params']
    return {
        'name': name,
        'params': params
    }


def get_lr_config(lr_adjuster):
    name = utils.general.get_class_name(lr_adjuster)
    params = dict(vars(lr_adjuster))
    params = remove_large_items(params)
    return {
        'name': name,
        'params': params
    }


def get_criterion_config(criterion):
    name = utils.general.get_class_name(criterion)
    return {
        'name': name
    }


def get_transforms_config(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__),
                         str(r.__dict__)))
    return data_aug


def get_visualizers_config(visualizers):
    return [v.classname for v in visualizers]


def get_metrics_config(metrics):
    return [m.name for m in metrics]


def get_aux_metrics_config(aux_metrics):
    return [m.__dict__ for m in aux_metrics]


