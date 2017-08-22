import os
import shutil
import time
import logging
from os.path import join

import config as cfg
import constants as c
from metrics import metric_builder
from visualizers import vis_utils
from notifications import emailer
import utils.logger
import training
import models.utils

from .exp_history import ExperimentHistory
from . import exp_utils
from . import exp_config




class Experiment():
    def __init__(self, name, parent_dir):
        self.name = name
        self.parent_dir = parent_dir
        self.root = join(parent_dir, name)
        self.weights_dir = join(self.root, 'weights')
        self.results_dir = join(self.root, 'results')
        self.history_dir = join(self.root, 'history')
        self.config_fpath = join(self.root, c.EXPERIMENT_CONFIG_FNAME)
        self.model_fpath = join(self.root, c.MODEL_FNAME)
        self.optim_fpath = join(self.root, c.OPTIM_FNAME)

        # Initialized/loaded later
        self.config = None
        self.logger = None
        self.history = None
        self.model = None
        self.optim = None
        self.max_patience = None
        self.early_stop_metric = None
        self.epoch = 0
        self.best_epoch = 1
        self.best_epoch_value = None
        self.visualizers = []
        self.metrics = []
        self.aux_metrics = []
        self.best_metrics = None

    def init(self, config_dict):
        self.config = exp_config.create_config_from_dict(config_dict)
        self.config.progress['status'] = c.INITIALIZED
        self.metrics = config_dict['metrics']
        self.aux_metrics = config_dict['aux_metrics']
        self.model = config_dict['model']
        self.optim = config_dict['optimizer']
        self.visualizers = config_dict['visualizers']
        self.max_patience = self.config.training['max_patience']
        self.early_stop_metric = self.config.training['early_stop_metric']
        self.history = ExperimentHistory(self.name, self.history_dir,
            self.metrics, self.aux_metrics)
        self.init_dirs()
        self.history.init()
        self.init_logger()
        self.init_visualizers()
        self.save_components()
        self.model.logger = self.logger

    def resume(self, epoch=None, verbose=False):
        self.init_logger()
        self.log("Resuming existing experiment")
        self.config = exp_config.load_config_from_file(self.config_fpath)
        self.config.progress['status'] = c.RESUMED
        self.load(verbose)
        self.init_visualizers()
        self.load_components(epoch)
        self.model.logger = self.logger

    def review(self, download=False, verbose=True):
        self.init_logger()
        if download:
            self.config = exp_config.fetch_external_config(self.name)
        else:
            self.config = exp_config.load_config_from_file(self.config_fpath)
        self.load(verbose=verbose)

    def init_visualizers(self):
        for v in self.visualizers:
            v.init(self.config)

    def init_dirs(self):
        os.makedirs(self.weights_dir)
        os.makedirs(self.history_dir)
        os.makedirs(self.results_dir)

    def init_logger(self, log_level=logging.INFO):
        self.logger = utils.logger.get_logger(
              self.root, 'logger', ch_log_level=log_level,
              fh_log_level=log_level)

    def load(self, verbose=False):
        self.metrics = metric_builder.get_metrics_from_config(self.config)
        self.aux_metrics = metric_builder.get_aux_metrics_from_config(self.config)
        self.visualizers = vis_utils.get_visualizers_from_config(self.config)
        self.history = ExperimentHistory(self.name, self.history_dir,
                            self.metrics, self.aux_metrics)
        self.history.resume()
        self.max_patience = self.config.training['max_patience']
        self.early_stop_metric = self.config.training['early_stop_metric']
        self.epoch = self.config.progress['epoch']
        self.best_metrics = self.config.progress['best_metrics']
        self.best_epoch = self.best_metrics[self.early_stop_metric]['epoch']
        self.best_epoch_value = self.best_metrics[self.early_stop_metric]['value']
        if verbose: self.config.summary(self.logger)

    def save(self, s3=cfg.S3_ENABLED, es=cfg.ES_ENABLED):
        self.config.save(s3, es)
        self.history.save(self.config, s3, es)

    def upload(self):
        exp_utils.upload_experiment(self.parent_dir, self.name)

    def load_components(self, epoch):
        self.model = models.utils.load_model(self.model_fpath)
        self.optim = training.load_optim(self.optim_fpath)
        self.load_model_state(epoch)
        self.load_optim_state(epoch)

    def save_components(self):
        models.utils.save_model(self.model.cpu(), self.model_fpath)
        training.save_optim(self.optim, self.optim_fpath)
        self.model = self.model.cuda()

    def log(self, msg):
        self.logger.info(msg)

    def update_visualizers(self, msg=None):
        for v in self.visualizers:
            v.update(self.config, self.history, msg)

    def update_progress(self):
        best = self.history.best_metrics
        self.best_epoch = best[self.early_stop_metric]['epoch']
        self.best_epoch_value = best[self.early_stop_metric]['value']
        self.config.progress['epoch'] = self.epoch
        self.config.progress['best_metrics'] = best

    def get_weights_fpath(self, epoch=None):
        fname = exp_utils.get_weights_fname(epoch)
        return join(self.weights_dir, fname)    

    def get_optim_fpath(self, epoch=None):
        fname = exp_utils.get_optim_fname(epoch)
        return join(self.weights_dir, fname)    

    def save_model_state(self, save_now=False):
        models.utils.save_weights(self.model, self.get_weights_fpath(), 
            epoch=self.epoch, name=self.name)
        if (save_now or self.epoch
            % self.config.training['save_weights_cadence'] == 0):
            fpath = self.get_weights_fpath(self.epoch)
            shutil.copyfile(self.get_weights_fpath(), fpath)

    def load_model_state(self, epoch=None):
        fpath = self.get_weights_fpath(epoch)
        models.utils.load_weights(self.model, fpath)

    def save_optim_state(self, save_now=False):
        training.save_optim_params(self.optim, self.get_optim_fpath(), 
            epoch=self.epoch, name=self.name)
        if (save_now or self.epoch
            % self.config.training['save_weights_cadence'] == 0):
            fpath = self.get_optim_fpath(self.epoch)
            shutil.copyfile(self.get_optim_fpath(), fpath)

    def load_optim_state(self, epoch=None):
        fpath = self.get_optim_fpath(epoch)
        training.load_optim_params(self.optim, fpath)

    def train(self, trainer, trn_loader, val_loader, n_epochs=None):
        start_epoch = self.epoch + 1 # Epochs start at 1
        self.config.progress['status'] = c.IN_PROGRESS
        self.config.progress['status_msg'] = 'Experiment in progress'

        if n_epochs is None:
            end_epoch = self.config.training['n_epochs'] + 1
        else:
            end_epoch = start_epoch + n_epochs
        try:
            for epoch in range(start_epoch, end_epoch):

                ### Adjust Lr ###
                lr_params = {'best_iter' : self.best_epoch}
                if trainer.lr_adjuster.iteration_type == 'epoch':
                    trainer.lr_adjuster.adjust(self.optim, epoch, lr_params)
                current_lr = trainer.lr_adjuster.get_learning_rate(self.optim)

                ### Train ###
                trn_start_time = time.time()
                trn_metrics = trainer.train(self.model, trn_loader, 
                    self.config.training['threshold'], epoch, self.metrics)
                trn_msg = training.log_trn_msg(self.logger, trn_start_time,
                                                trn_metrics, current_lr, epoch)

                ### Test ###
                val_start_time = time.time()
                val_metrics = trainer.test(self.model, val_loader, 
                    self.config.training['threshold'], self.metrics)
                val_msg = training.log_val_msg(self.logger, val_start_time,
                                                val_metrics, current_lr)

                sys_mem = training.log_memory('')

                ### Save Metrics ###
                aux_metrics = [current_lr, sys_mem]
                self.history.save_metric(c.TRAIN, trn_metrics, epoch)
                self.history.save_metric(c.VAL, val_metrics, epoch)
                self.history.save_aux_metrics(aux_metrics, epoch)
                self.history.update_best_metrics()

                ### Checkpoint ###
                self.epoch = epoch
                self.update_progress()
                self.save()
                self.save_model_state()
                self.save_optim_state()
                self.update_visualizers('\n'.join([trn_msg, val_msg]))

                ### Early Stopping ###
                if training.early_stop(epoch, self.best_epoch, self.max_patience):
                    msg = "Early stopping at epoch %d since no better %s found since epoch %d at %.3f" % (
                        epoch, self.early_stop_metric, self.best_epoch, self.best_epoch_value)
                    self.config.progress['status'] = c.MAX_PATIENCE_EXCEEDED
                    self.config.progress['status_msg'] = msg
                    break

        except Exception as e:
            self.config.progress['status'] = c.FAILED
            self.config.progress['status_msg'] = e
            raise Exception(e)
        finally:
            if self.config.progress['status'] == c.IN_PROGRESS:
                self.config.progress['status'] = c.COMPLETED
                self.config.progress['status_msg'] = 'Experiment Complete!'
            if cfg.EMAIL_ENABLED:
                emailer.send_experiment_status_email(self, cfg.USER_EMAIL)
            self.log(self.config.progress['status_msg'])


