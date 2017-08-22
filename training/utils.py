import gc
import objgraph
import resource
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math

from predictions import pred_utils
import constants as c



def train_model(model, dataloader, thresholds, optimizer, criterion,
                lr_adjuster, epoch, n_epochs, metrics=[]):
    model.train()
    n_batches = len(dataloader)
    cur_iter = int((epoch-1) * n_batches)+1
    total_iter = int(n_batches * n_epochs)
    metric_totals = {m.name:0 for m in metrics}

    for inputs, targets, img_paths in dataloader:
        if len(targets.size()) == 1:
            targets = targets.float().view(-1, 1)
        inputs = Variable(inputs.cuda(async=True))
        targets = Variable(targets.cuda(async=True))

        ## Forward Pass
        output = model(inputs)

        ## Clear Gradients
        model.zero_grad()

        # Metrics
        loss = criterion(output, targets)
        loss_data = loss.data[0]
        labels = targets.data.cpu().numpy()
        probs = output.data.cpu().numpy()
        preds = pred_utils.get_predictions(probs, thresholds)

        for metric in metrics:
            score = metric.evaluate(loss_data, preds, probs, labels)
            metric_totals[metric.name] += score

        ## Backprop
        loss.backward()
        optimizer.step()

        ### Adjust Lr ###
        if lr_adjuster.iteration_type == 'mini_batch':
            lr_adjuster.adjust(optimizer, cur_iter)
        cur_iter += 1

    for metric in metrics:
        metric_totals[metric.name] /= n_batches

    return metric_totals


def test_model(model, loader, thresholds, n_classes, criterion, metrics):
    model.eval()

    loss = 0
    probs = np.empty((0, n_classes))
    labels = np.empty((0, n_classes))
    metric_totals = {m.name:0 for m in metrics}

    for inputs, targets, img_paths in loader:
        if len(targets.size()) == 1:
            targets = targets.float().view(-1,1)
        inputs = Variable(inputs.cuda(async=True), volatile=True)
        targets = Variable(targets.cuda(async=True), volatile=True)

        output = model(inputs)

        loss += criterion(output, targets).data[0]
        probs = np.vstack([probs, output.data.cpu().numpy()])
        labels = np.vstack([labels, targets.data.cpu().numpy()])

    loss /= len(loader)
    preds = pred_utils.get_predictions(probs, thresholds)
    for metric in metrics:
        score = metric.evaluate(loss, preds, probs, labels)
        metric_totals[metric.name] = score

    return metric_totals


def early_stop(epoch, best_epoch, patience):
    return (epoch - best_epoch) > patience


def log_memory(step):
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print("Memory usage ({:s}): {:.2f} MB\n".format(step, max_mem_used))
    return max_mem_used


def log_trn_msg(logger, start_time, trn_metrics, lr, epoch):
    epoch_msg = 'Epoch {:d}'.format(epoch)
    metric_msg = get_metric_msg(logger, c.TRAIN, trn_metrics, lr)
    time_msg = get_time_msg(start_time)
    combined = epoch_msg + '\n' + metric_msg + time_msg
    logger.info(combined)
    return combined


def log_val_msg(logger, start_time, trn_metrics, lr):
    metric_msg = get_metric_msg(logger, c.VAL, trn_metrics, lr)
    time_msg = get_time_msg(start_time)
    combined = metric_msg + time_msg
    logger.info(combined)
    return combined


def get_metric_msg(logger, dset, metrics_dict, lr=0):
    msg = dset.capitalize() + ' - '
    for name in metrics_dict.keys():
        metric_str = ('{:.4f}').format(metrics_dict[name]).lstrip('0')
        msg += ('{:s} {:s} | ').format(name, metric_str)
    msg += 'LR ' + '{:.6f}'.format(lr).rstrip('0').lstrip('0') + ' | '
    return msg


def get_time_msg(start_time):
    time_elapsed = time.time() - start_time
    msg = 'Time {:.1f}m {:.2f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    return msg


def load_optim_params(optim, fpath):
    state = torch.load(fpath)
    optim.load_state_dict(state['state_dict'])


def save_optim_params(optim, fpath, epoch=None, name=None):
    torch.save({
        'name': name,
        'epoch': epoch,
        'state_dict': optim.state_dict()
    }, fpath)


def load_optim(fpath):
    return torch.load(fpath)


def save_optim(optim, fpath):
    torch.save(optim, fpath)
