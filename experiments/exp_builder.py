from glob import glob
import pandas as pd
from .experiment import Experiment



def prune_experiments(exp_dir, exp_names):
    # Delete weights except for best weight in `n_bins`
    for name in exp_names:
        exp = Experiment(name, exp_dir)
        exp.review(verbose=False)
        exp.auto_prune(n_bins=5, metric_name='Loss', func=max)


def build_exp_summary_dict(exp):
    config = exp.config
    history = exp.history
    dict_ = {
        'name': config.name,
        'exp_id': config.get_id(),
        'created': config.created,
        'fold': config.data['dset_fold'],
        'model_name' : config.model_name,
        'threshold' : config.training['threshold'],
        'model_name' : config.model['name'],
        'optim' : config.optimizer['name'],
        'optim_params' : str(config.optimizer['params']),
        'lr_adjuster' : config.lr_adjuster['name'],
        'lr_adjuster_params' : str(config.lr_adjuster['params']),
        'criterion': config.criterion['name'],
        'transforms' : ', '.join([t[0] for t in config.transforms]),
        ### initial lr, img_scale, rescale, total_epochs
        'transforms' : ', '.join([t[0] for t in config.transforms]),
        'init_lr':config.training['initial_lr'],
        'wdecay':config.training['weight_decay'],
        'batch': config.training['batch_size'],
        'img_scl':config.data['img_scale'],
        'img_rescl': config.data['img_rescale'],
        'nepochs':config.training['n_epochs'],
    }
    for name in config.metrics:
        dict_[name+'Epoch'] = history.best_metrics[name]['epoch']
        dict_[name+'Val'] = history.best_metrics[name]['value']
    return dict_


def build_exps_df_from_dir(exps_dir):
    exp_names = glob(exps_dir+'/*/')
    summaries = []
    for name in exp_names:
        exp = Experiment(name, exps_dir)
        exp.review(verbose=False)
        exp_summary = build_exp_summary_dict(exp)
        summaries.append(exp_summary)
    return pd.DataFrame(summaries)


def upload_experiments(exp_dir):
    exp_paths = glob(exp_dir+'/*/')
    for path in exp_paths:
        name = path.strip('/').split('/')[-1]
        exp = Experiment(name, exp_dir)
        exp.review(verbose=False)
        exp.save()


def upload_experiments(exp_dir):
    exp_paths = glob(exp_dir+'/*/')
    for path in exp_paths:
        name = path.strip('/').split('/')[-1]
        exp = Experiment(name, exp_dir)
        exp.review(verbose=False)
        exp.save()
