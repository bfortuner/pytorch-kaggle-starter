import numpy as np
from visdom import Visdom
import constants as c

class Viz():

    def __init__(self, exp_name):
        self.name = exp_name
        self.classname = 'Visdom'
        self.viz = None
        self.plots = None

    def init(self, exp_config):
        self.viz = Visdom()
        self.plots = self.init_visdom_plots(exp_config)

    def update(self, exp_config, exp_history, msg=None):
        epoch = exp_config.progress['epoch']
        metrics_history = exp_history.metrics_history

        for name in exp_config.metrics:
            trn_arr = np.array(metrics_history[name][c.TRAIN])
            val_arr = np.array(metrics_history[name][c.VAL])
            self.update_metric_plot(name, trn_arr, val_arr, epoch,
                                    ylabel=name)

        for metric in exp_config.aux_metrics:
            name = metric['name']
            data_arr = np.array(metrics_history[name])
            self.update_aux_metric_plot(name, data_arr, epoch,
                                        ylabel=metric['units'])
        self.update_summary_plot(msg)

    def init_visdom_plots(self, exp_config):
        plots = {}
        for name in exp_config.metrics:
            plot = self.init_train_val_metric_plot(name, name)
            plots[name] = plot
        for aux_metric in exp_config.aux_metrics:
            name = aux_metric['name']
            plot = self.init_auxiliary_metric_plot(name, aux_metric['units'])
            plots[name] = plot
        plots['summary'] = self.init_txt_plot('summary')
        return plots

    def init_train_val_metric_plot(self, title, ylabel, xlabel='epoch'):
        return self.viz.line(
            X=np.array([1]),
            Y=np.array([[1, 1]]),
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                legend=['Train', 'Valid']
            ),
            env=self.name
        )

    def init_auxiliary_metric_plot(self, title, ylabel, xlabel='epoch'):
        return self.viz.line(
            X=np.array([1]),
            Y=np.array([1]),
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                legend=[]
            ),
            env=self.name
        )

    def init_txt_plot(self, title):
        return self.viz.text(
            "Initializing.. " + title,
            env=self.name
        )

    def viz_epochs(self, cur_epoch):
        # Epochs start at 1
        epochs = np.arange(1, cur_epoch+1)
        return np.stack([epochs, epochs],1)

    def update_metric_plot(self, metric, train_arr, val_arr,
                           epoch, ylabel, xlabel='epoch'):
        data = np.stack([train_arr, val_arr], 1)
        window = self.plots[metric]
        return self.viz.line(
            X=self.viz_epochs(epoch),
            Y=data,
            win=window,
            env=self.name,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=metric,
                legend=['Train', 'Valid']
            ),
        )

    def update_aux_metric_plot(self, metric, data_arr, epoch, ylabel,
                                   xlabel='epoch', legend=[]):
        window = self.plots[metric]
        return self.viz.line(
            X=self.viz_epochs(epoch)[:,0],
            Y=data_arr,
            win=window,
            env=self.name,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel, #metric.units,
                title=metric,
                legend=legend
            ),
        )

    def update_summary_plot(self, msg):
        window = self.plots['summary']
        return self.viz.text(
            msg,
            win=window,
            env=self.name
        )


def load(config):
    return Viz(config.name)
