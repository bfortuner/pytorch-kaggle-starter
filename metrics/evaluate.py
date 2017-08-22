import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as scipy_metrics
import sklearn
import itertools

import utils
import predictions
import datasets.metadata as meta
import metrics
import constants as c



def get_evaluate_df(preds, probs, targets, fpaths, label_names):
    fnames = utils.files.get_fnames_from_fpaths(fpaths)
    preds_df = pd.DataFrame(preds, columns=label_names, dtype=int)
    probs_df = pd.DataFrame(np.round(probs, 2),
                        columns=['p_'+l for l in label_names], dtype=float)
    targets_df = pd.DataFrame(targets,
                        columns=['t_'+l for l in label_names], dtype=int)
    evaluate_df = pd.concat([preds_df, probs_df, targets_df], axis=1)
    evaluate_df.insert(len(evaluate_df.columns),'fpath',
                    pd.Series(fpaths, index=evaluate_df.index))
    evaluate_df.insert(0,'fname', pd.Series(fnames, index=evaluate_df.index))
    return evaluate_df


def get_preds_by_target_label(df, label, outcome='all', condensed=False, 
                              shuffle=True):
    t = 't_'+label.lower()
    p = 'p_'+label.lower()
    if outcome == 'all':
        label_preds = df[df[t] == 1]
    elif outcome == 'correct':
        label_preds = df[df[t] == df[label]]
    elif outcome == 'incorrect':
        label_preds = df[df[t] != df[label]]
    elif outcome == 'TP':
        label_preds = df[(df[t] == 1) & (df[label] == 1)]
    elif outcome == 'TN':
        label_preds = df[(df[t] == 0) & (df[label] == 0)]
    elif outcome == 'FP':
        label_preds = df[(df[t] == 0) & (df[label] == 1)]
    elif outcome == 'FN':
        label_preds = df[(df[t] == 1) & (df[label] == 0)]
    if condensed:
        label_preds = label_preds[[label, p, t, 'fpath']]
    if shuffle:
        return label_preds.sample(frac=1)
    return label_preds


def get_preds_by_predicted_label(df, label, condensed=False, shuffle=True):
    t = 't_'+label.lower()
    p = 'p_'+label.lower()
    label_preds = df[df[label] == 1][[label, p, t, 'fname']]
    if condensed:
        label_preds = label_preds[[label, p, t, 'fpath']]
    if shuffle:
        return label_preds.sample(frac=1)
    return label_preds


def get_preds_by_target_and_prob(df, label, targ, p_min=0.0, p_max=1.0,
                                 shuffle=True):
    t = 't_'+label.lower()
    p = 'p_'+label.lower()
    label_preds = df[(df[t] == targ) & (df[p] >= p_min) & (df[p] <= p_max)]
    if shuffle:
        return label_preds.sample(frac=1)
    return label_preds


def plot_pred_from_df_idx(df, idx, label_names, fs=(5,5)):
    img_row = df.loc[idx]
    title = get_img_title_for_plot(df, idx, label_names)
    img_utils.plot_img_from_fpath(img_row['img_path'], fs=fs, title=title)


def get_img_title_for_plot(df, idx, label_names):
    img_row = df.loc[idx]
    pred_headers = label_names
    prob_headers = ['p_'+l for l in label_names]
    target_headers = ['t_'+l for l in label_names]
    pred_tag = meta.convert_one_hot_to_tags(
                    np.array(img_row[pred_headers]), label_names)
    target_tag = meta.convert_one_hot_to_tags(
                    np.array(img_row[target_headers]), label_names)
    prob_targ_pct = meta.convert_one_hot_to_tags(
                    np.array(img_row[target_headers]),
                    np.array(img_row[prob_headers]).astype(float))
    prob_pred_pct = meta.convert_one_hot_to_tags(
                    np.array(img_row[pred_headers]),
                    np.array(img_row[prob_headers]).astype(float))
    prob_targ_pct = np.round(prob_targ_pct, 3)
    prob_pred_pct = np.round(prob_pred_pct, 3)
    title = ("Trg: " + str(target_tag) + "\nPrb: " + str(prob_targ_pct) +
             "\nPrd: " + str(pred_tag) + "\nPrb: " + str(prob_pred_pct) +
             '\n' + img_row['fname'].split('/')[-1])
    return title


def plot_predictions(df, label_names, n=6, rows=3, cols=3, fs=(20,16)):
    plt.figure(figsize=fs)
    j = 1
    for idx, row in df.iterrows():
        plt.subplot(rows, cols, j)
        plt.imshow(plt.imread(row['fpath']))
        title = get_img_title_for_plot(df, idx, label_names)
        plt.title(title)
        plt.axis('off')
        j+=1
        if j > n:
            break


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Target')
    plt.xlabel('Prediction')


def plot_label_confusion_matrix(df, label):
    cm = sklearn.metrics.confusion_matrix(df['t_'+label], df[label])
    title = label.upper()
    plot_confusion_matrix(cm, {'not present':0, 'present':1}, title=title)


def plot_label_level_cms(df, label_names):
    for label in label_names:
        plot_label_confusion_matrix(df, label)


def plot_roc_curve(probs, targets):
    fpr, tpr, thresholds = scipy_metrics.roc_curve(
                targets.flatten(), probs.flatten())
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)



# Multi-label methods

def get_samples_containing_labels(df, labels, target, sample_by='target',
                                  shuffle=True):
    if sample_by == 'target':
        prefix = 't_'
    else:
        prefix = ''
    query = ''
    for label in labels[:-1]:
        query +=  prefix + label + ' == '+str(target)+' & '
    query += prefix + labels[-1] + ' == '+str(target)
    preds = df.query(query)
    if shuffle:
        return preds.sample(frac=1)
    return preds


def get_summary_metrics_by_labels(df, labels):
    query = ''
    for label in labels[:-1]:
        query +=  "lb == '"+label+"' | "
    query += "lb == '"+labels[-1]+"'"
    metrics = df.query(query)
    return metrics


def get_label_freq_bins(labels, label_names):
    indices = np.arange(len(label_names))
    binned_labels = np.bincount(labels, minlength=len(label_names))
    return np.column_stack([indices, binned_labels])


def graph_summary_metric(summary_df, metric_name, sort_desc=True):
    plt_df = summary_df.loc[:,['lb',metric_name]][:-2].sort_values(
                           [metric_name], ascending=sort_desc)
    plt_df[['lb']]
    myplot = plt_df.plot(kind='barh',title=metric_name, figsize=(10,6))
    myplot.set_yticklabels(plt_df.lb.values)
    myplot.set_xlabel(metric_name)
    myplot.set_ylabel('label')
    plt.show()


def get_label_probs(df, label, targ, p_range, bins=15):
    t = 't_'+label.lower()
    p = 'p_'+label.lower()
    label_preds = df[(df[t] == targ) & (df[p] >= p_range[0]) & (df[p] <= p_range[1])]
    return label_preds['p_'+label]


def plot_label_level_prob_hists(df, label_name, targ=1, pred=None, prob=1.0):
    for label in label_names:
        get_label_prob_hist(df, label, targ, pred, prob)


def get_multi_label_summary_metrics(preds, probs, targets, label_names, verbose=True):
    """
    Currently designed for multi-label classification
    """
    label_level_accuracy = np.round(metrics.get_accuracy(
                                        preds, targets),3)
    img_level_accuracy = np.round(scipy_metrics.accuracy_score(
                                        targets, preds),3)
    correct_img_idx, correct_label_idx = np.where(preds==targets)
    incorrect_img_idx, incorrect_label_idx = np.where(preds!=targets)

    accuracy = metrics.get_accuracy(preds, targets)
    error = np.sum(preds!=targets) / len(preds.flatten())
    f2_score = metrics.get_f2_score(preds, targets, 'samples')

    # TP/FP/TN/FN
    TP_img_idx, TP_label_idx = np.where((preds==targets) & (preds==1))
    FP_img_idx, FP_label_idx = np.where((preds!=targets) & (preds==1))
    TN_img_idx, TN_label_idx = np.where((preds==targets) & (preds==0))
    FN_img_idx, FN_label_idx = np.where((preds!=targets) & (preds==0))
    TP,FP,TN,FN = TP_label_idx,FP_label_idx,TN_label_idx,FN_label_idx
    n_TP = len(TP_label_idx)
    n_FP = len(FP_label_idx)
    n_TN = len(TN_label_idx)
    n_FN = len(FN_label_idx)

    #Labels
    n_labels = len(preds.flatten())
    correct_labels_cnt = np.count_nonzero(preds==targets)
    incorrect_labels_cnt = np.count_nonzero(preds!=targets)
    assert (correct_labels_cnt+incorrect_labels_cnt == n_labels)

    # Total Positive/True/One Labels
    total_positive_labels = np.sum(targets)
    total_positive_labels_by_class = np.sum(targets, axis=0)

    #Images
    n_imgs = len(preds)
    image_idx = np.unique(np.where(preds==targets))
    incorrect_images_idx = np.unique(incorrect_img_idx)
    mask = np.in1d(image_idx, incorrect_images_idx)
    correct_images_idx = np.where(~mask)[0]
    n_imgs_correct = len(correct_images_idx)
    n_imgs_incorrect = len(incorrect_images_idx)
    assert (n_imgs_correct+n_imgs_incorrect == n_imgs)

    correct_freq = get_label_freq_bins(correct_label_idx, label_names)
    incorrect_freq = get_label_freq_bins(incorrect_label_idx, label_names)
    total_freq = correct_freq[:,1] + incorrect_freq[:,1]
    total_ones = np.sum(targets, axis=0)
    percent_ones = np.round(total_ones/total_freq*100,1)
    assert np.sum(incorrect_freq[:,1]) + np.sum(
                correct_freq[:,1]) == n_labels

    # Truth
    tp_freq = get_label_freq_bins(TP_label_idx, label_names)
    fp_freq = get_label_freq_bins(FP_label_idx, label_names)
    tn_freq = get_label_freq_bins(TN_label_idx, label_names)
    fn_freq = get_label_freq_bins(FN_label_idx, label_names)
    assert np.sum(tp_freq[:,1]) == n_TP
    assert np.sum(fp_freq[:,1]) == n_FP
    assert np.sum(tn_freq[:,1]) == n_TN
    assert np.sum(fn_freq[:,1]) == n_FN

    # Metrics
    error_pct = np.round(incorrect_freq[:,1] / total_freq * 100,1)
    weighted_error_pct = np.round(incorrect_freq[:,1]/np.sum(
        incorrect_freq[:,1]),2)
    #http://ml-cheatsheet.readthedocs.io/en/latest/glossary.html?highlight=precision
    total_precision = n_TP/(n_TP+n_FP)
    total_recall = n_TP/(n_TP+n_FN)
    precision_by_label = np.round(
        tp_freq[:,1]/(tp_freq[:,1]+fp_freq[:,1])*100,1)
    recall_by_label = np.round(
        tp_freq[:,1]/(tp_freq[:,1]+fn_freq[:,1])*100,1)
    weighted_fp_pct = np.round(fp_freq/n_FP*100,1)[:,1]
    weighted_fn_pct = np.round(fn_freq/n_FN*100,1)[:,1]
    mean_prob_by_label = np.round(np.mean(probs, axis=0),2)
    median_prob_by_label = np.round(np.median(probs, axis=0),2)

    combined_pivot = np.column_stack([error_pct,
                                    weighted_error_pct,
                                    precision_by_label,
                                    recall_by_label,
                                    correct_freq[:,1],
                                    incorrect_freq[:,1],
                                    tp_freq[:,1],
                                    tn_freq[:,1],
                                    fp_freq[:,1],
                                    fn_freq[:,1],
                                    weighted_fp_pct,
                                    weighted_fn_pct,
                                    total_ones,
                                    percent_ones,
                                    mean_prob_by_label,
                                    median_prob_by_label])

    columns = [
    'err_pct','wt_err_pct', 'precision','recall',
    'correct_labels','incorrect_labels','tp','tn', 'fp','fn',
    'wt_fp_pct','wt_fn_pct','total_ones','pct_ones','mean_prb','med_prb'
    ]
    int_columns = ['total_ones','correct_labels','incorrect_labels',
        'tp','tn','fp','fn'
    ]
    float_columns = ['pct_ones','err_pct','precision','recall']
    combined_pivot[np.isnan(combined_pivot)] = 0
    summary_df = pd.DataFrame(combined_pivot, columns=columns)
    summary_df.insert(0, 'lb', pd.Series(
        label_names, index=summary_df.index))
    # sum_row = summary_df.sum(numeric_only=True)
    # sum_row['lb'] = 'sum'
    # mean_row = np.round(summary_df.mean(numeric_only=True), 1)
    # mean_row['lb'] = 'mean'
    # summary_df = summary_df.append(sum_row, ignore_index=True)
    # summary_df = summary_df.append(mean_row, ignore_index=True)
    summary_df[int_columns] = summary_df[int_columns].astype(int)

    if verbose:
        print("Error", round(error, 4),"\nAcc",round(accuracy, 4),
              "\nn_labels",n_labels,"\nn_labels_correct",correct_labels_cnt,
              "\nn_labels_incorrect",incorrect_labels_cnt,
              "\nn_imgs",n_imgs, "\nn_imgs_correct", n_imgs_correct,
              "\nn_imgs_incorrect", n_imgs_incorrect, '\ntotal_one_labels',
               total_positive_labels, '\nlabel_level_accuracy',
               label_level_accuracy,'\nimg_level_accuracy',img_level_accuracy)

    return summary_df
