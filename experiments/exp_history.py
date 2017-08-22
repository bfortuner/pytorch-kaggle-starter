import os
import json
from os.path import join
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import utils.files
import pandas as pd
from io import StringIO
mpl.use('Agg')
plt.style.use('bmh')

import config as cfg
import constants as c
from clients import s3_client, es_client



class ExperimentHistory():

    def __init__(self, exp_name, history_dir, metrics=None, aux_metrics=None):
        self.exp_name = exp_name
        self.history_dir = history_dir
        self.train_history_fpath = join(self.history_dir, c.TRAIN+'.csv')
        self.val_history_fpath = join(self.history_dir, c.VAL+'.csv')
        self.aux_metrics_fpath = join(self.history_dir, 'aux_metrics.csv')
        self.summary_fpath = join(self.history_dir, exp_name+'.csv')
        self.metrics = metrics
        self.aux_metrics = aux_metrics
        self.metrics_history = None
        self.best_metrics = {}

    def init(self):
        self.init_metrics()
        self.init_history_files()

    def resume(self, fetch=False):
        self.init_metrics()
        if fetch:
            self.load_from_external()
        else:
            self.load_from_files()
        self.update_best_metrics()

    def init_history_files(self):
        Path(self.train_history_fpath).touch()
        Path(self.val_history_fpath).touch()
        Path(self.aux_metrics_fpath).touch()

    def init_metrics(self):
        histories = {}
        for metric in self.metrics:
            histories[metric.name] = {
                c.TRAIN: [],
                c.VAL: []
            }
        for aux_metric in self.aux_metrics:
            histories[aux_metric.name] = []
        self.metrics_history = histories

    def save(self, config, s3=cfg.S3_ENABLED, es=cfg.S3_ENABLED):
        df = pd.DataFrame()
        for metric in self.metrics:
            trn_data = self.metrics_history[metric.name][c.TRAIN]
            val_data = self.metrics_history[metric.name][c.VAL]
            df[c.TRAIN+'_'+metric.name] = trn_data
            df[c.VAL+'_'+metric.name] = val_data

        for aux_metric in self.aux_metrics:
            df[aux_metric.name] = self.metrics_history[aux_metric.name]

        epochs = pd.Series([i for i in range(1,len(trn_data)+1)])
        df.insert(0, 'Epoch', epochs)
        df.to_csv(self.summary_fpath, index=False)

        if s3:
            s3_client.upload_experiment_history(self.summary_fpath,
                                                self.exp_name)
        if es:
            es_client.upload_experiment_history(config, self)

    def load_from_files(self):
        self.load_history_from_file(c.TRAIN)
        self.load_history_from_file(c.VAL)
        self.load_aux_metrics_from_file()

    def load_from_external(self):
        df = self.fetch_dataframe()
        for metric in self.metrics:
            for dset in [c.TRAIN, c.VAL]:
                data = df[dset+'_'+metric.name].tolist()
                self.metrics_history[metric.name][dset] = data
        for aux_metric in self.aux_metrics:
            data = df[aux_metric.name].tolist()
            self.metrics_history[aux_metric.name] = data

    def get_dataframe(self):
        if os.path.isfile(self.summary_fpath):
            return self.load_dataframe_from_file()
        return self.fetch_dataframe()

    def fetch_dataframe(self):
        csv_str = s3_client.fetch_experiment_history(self.exp_name)
        df = pd.DataFrame
        data = StringIO(csv_str)
        return pd.read_csv(data, sep=",")

    def load_dataframe_from_file(self):
        df = pd.read_csv(self.summary_fpath, sep=',')
        return df

    def save_metric(self, dset_type, values_dict, epoch):
        values_arr = []
        for metric in self.metrics:
            value = values_dict[metric.name]
            self.metrics_history[metric.name][dset_type].append(value)
            values_arr.append(value)
        fpath = join(self.history_dir, dset_type+'.csv')
        self.append_history_to_file(fpath, values_arr, epoch)

    def load_history_from_file(self, dset_type):
        fpath = join(self.history_dir, dset_type+'.csv')
        data = np.loadtxt(fpath, delimiter=',').reshape(
                            -1, len(self.metrics)+1)
        for i in range(len(self.metrics)):
            self.metrics_history[self.metrics[i].name][dset_type] = data[:,i+1].tolist()

    def append_history_to_file(self, fpath, values, epoch):
        # Restricts decimals to 6 places!!!
        formatted_vals = ["{:.6f}".format(v) for v in values]
        line = ','.join(formatted_vals)
        with open(fpath, 'a') as f:
            f.write('{},{}\n'.format(epoch, line))

    def update_best_metrics(self):
        best_metrics = {}
        for metric in self.metrics:
            metric_history = self.metrics_history[metric.name][c.VAL]
            best_epoch, best_value = metric.get_best_epoch(
                metric_history)
            best_metrics[metric.name] = {
                'epoch':best_epoch,
                'value':best_value
            }
        self.best_metrics = best_metrics

    def load_aux_metrics_from_file(self):
        data = np.loadtxt(self.aux_metrics_fpath, delimiter=',').reshape(
               -1, len(self.aux_metrics)+1)
        for i in range(len(self.aux_metrics)):
            self.metrics_history[self.aux_metrics[i].name] = data[:,i+1].tolist()

    def save_aux_metrics(self, values, epoch):
        for i in range(len(self.aux_metrics)):
            self.metrics_history[self.aux_metrics[i].name].append(values[i])
        self.append_history_to_file(self.aux_metrics_fpath, values, epoch)

    def get_dset_arr(self, dset):
        data = []
        for metric in self.metrics:
            data.append(self.metrics_history[metric.name][dset])
        epochs = [i+1 for i in range(len(data[0]))]
        data.insert(0,epochs)
        arr = np.array(data)
        return arr.T

    def plot(self, save=False):
        trn_data = self.get_dset_arr(c.TRAIN)
        val_data = self.get_dset_arr(c.VAL)
        metrics_idx = [i+1 for i in range(len(self.metrics))]
        trn_args = np.split(trn_data, metrics_idx, axis=1)
        val_args = np.split(val_data, metrics_idx, axis=1)

        metric_fpaths = []
        for i in range(len(self.metrics)):
            metric_trn_data = trn_data[:,i+1] #skip epoch
            metric_val_data = val_data[:,i+1]

            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            plt.plot(trn_args[0], metric_trn_data, label='Train')
            plt.plot(val_args[0], metric_val_data, label='Validation')
            plt.title(self.metrics[i].name)
            plt.xlabel('Epoch')
            plt.ylabel(self.metrics[i].name)
            plt.legend()
            ax.set_yscale('log')

            if save:
                metric_fpath = join(self.history_dir,
                                    self.metrics[i].name+'.png')
                metric_fpaths.append(metric_fpath)
                plt.savefig(metric_fpath)

        # Combined View
        if save:
            all_metrics_fpath = join(self.history_dir, 'all_metrics.png')
            metric_fpaths.append(all_metrics_fpath)
            os.system('convert +append ' + ' '.join(metric_fpaths))

        plt.show()

    def to_doc(self, config):
        df = self.get_dataframe()
        df[c.EXP_ID_FIELD] = config.get_id()
        df[c.ES_EXP_KEY_FIELD] = df['Epoch'].map(str) + '_' + config.get_id()
        df['name'] = config.get_display_name()
        df['user'] = config.hardware['hostname']
        df['criterion'] = config.criterion['name']
        df['optim'] = config.optimizer['name']
        df['init_lr'] = config.training['initial_lr']
        df['wd'] = config.training['weight_decay']
        df['bs'] = config.training['batch_size']
        df['imsz'] = config.data['img_rescale']
        df['model_name'] = config.model_name
        df['lr_adjuster'] = config.lr_adjuster['name']
        df['threshold'] = config.training['threshold']
        return json.loads(df.to_json(orient='records'))


    ### TODO
    def get_history_summary(self, epoch, early_stop_metric):
        msg = ['Epoch: %d' % epoch]
        for dset in [c.TRAIN, c.VAL]:
            dset_msg = dset.capitalize() + ' - '
            for metric in self.metrics:
                value = self.metrics_history[metric.name][dset][-1]
                dset_msg += '{:s}: {:.3f} '.format(metric.name, value)
            msg.append(dset_msg)

        best_epoch = self.best_metrics[early_stop_metric]['epoch']
        best_epoch_value = self.best_metrics[early_stop_metric]['value']
        best_metric_msg = 'Best val {:s}: Epoch {:d} - {:.3f}'.format(
            early_stop_metric, best_epoch, best_epoch_value)
        msg.append(best_metric_msg)

        return '\n'.join(msg)

