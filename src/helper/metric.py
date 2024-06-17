import numpy as np
from numpy import ndarray
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, recall_score


def offline_metric(labels, predicts, n_class: int):
    """
    :return: accuracy, gmean, precision, recall, fscore, support, confusion
    """
    labels = np.array(labels)
    predicts = np.array(predicts)
    confusion = confusion_matrix(labels, predicts, labels=np.arange(0, n_class))
    precision, recall, fscore, support = precision_recall_fscore_support(labels, predicts, labels=list(range(n_class)))
    accuracy = confusion.trace() / confusion.sum()
    gmean = recall.prod() ** (1 / n_class)
    return accuracy, gmean, precision, recall, fscore, support, confusion


def online_confusion_matrix(labels, predicts, fading_factor: float, n_class: int = None,
                            confusion_matrix: ndarray = None):
    if confusion_matrix is None:
        confusion_matrix = np.zeros(shape=(n_class, n_class))

    for label, predict in zip(labels, predicts):
        confusion_matrix = confusion_matrix * fading_factor
        confusion_matrix[label, predict] += 1
    return confusion_matrix


def precision_recall_fscore_support_by_confusion_matrix(confusion_matrix: ndarray):
    n_class = len(confusion_matrix)
    precision = np.zeros(n_class)
    recall = np.zeros(n_class)
    fscore = np.zeros(n_class)
    support = np.zeros(n_class)

    for i in range(n_class):
        true_positive = confusion_matrix[i, i]
        false_positive = confusion_matrix[:, i].sum() - true_positive
        false_negative = confusion_matrix[i, :].sum() - true_positive

        precision[i] = true_positive / (true_positive + false_positive + 1e-6)
        recall[i] = true_positive / (true_positive + false_negative + 1e-6)
        fscore[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-6)
        support[i] = confusion_matrix[i, :].sum()

    accuracy = confusion_matrix.trace() / confusion_matrix.sum()
    gmean = recall.prod()
    gmean = gmean ** (1 / n_class)
    return accuracy, gmean, precision, recall, fscore, support


def online_metric_deprecated(labels, predicts, fading_factor: float, n_class: int, confusion_matrix: ndarray = None):
    """
    Warning: this only gives the performance at the last time step, not the overall performance!
    :return: accuracy, gmean, precision, recall, fscore, support, confusion
    """
    confusion = online_confusion_matrix(labels, predicts, fading_factor=fading_factor, n_class=n_class,
                                        confusion_matrix=confusion_matrix)
    accuracy, gmean, precision, recall, fscore, support = precision_recall_fscore_support_by_confusion_matrix(confusion)
    return accuracy, gmean, precision, recall, fscore, support, confusion


def metric_by_confusion(labels, predicts, fading_factor: float, n_class: int, confusion_matrix: ndarray = None):
    """
    :return: accuracy, gmean, precision, recall, fscore, support, confusion
    """
    confusion = online_confusion_matrix(labels, predicts, fading_factor=fading_factor, n_class=n_class,
                                        confusion_matrix=confusion_matrix)
    accuracy, gmean, precision, recall, fscore, support = precision_recall_fscore_support_by_confusion_matrix(confusion)
    return accuracy, gmean, precision, recall, fscore, support, confusion


def online_metric(labels, predicts, fading_factor: float, n_class: int, confusion_matrix: ndarray = None,
                  overall_out: bool = False, latest_out: bool = False):
    """
    :return: accuracy, gmean, precision, recall, fscore, support, confusion
    """
    if confusion_matrix is None:
        confusion_matrix = np.zeros(shape=(n_class, n_class))

    accuracy_li, gmean_li, precision_li, recall_li, fscore_li, support_li, confusion_li = [], [], [], [], [], [], []
    for label, predict in zip(labels, predicts):
        confusion_matrix = confusion_matrix * fading_factor
        confusion_matrix[label, predict] += 1
        accuracy, gmean, precision, recall, fscore, support = precision_recall_fscore_support_by_confusion_matrix(
            confusion_matrix)
        accuracy_li.append(accuracy)
        gmean_li.append(gmean)
        precision_li.append(precision)
        recall_li.append(recall)
        fscore_li.append(fscore)
        support_li.append(support)
        confusion_li.append(confusion_matrix)
    performances = accuracy_li, gmean_li, precision_li, recall_li, fscore_li, support_li, confusion_li
    assert not (overall_out and latest_out)
    if overall_out:
        return nanmeans(*performances)
    elif latest_out:
        return latest(*performances)
    else:
        return performances


def nanmeans(*args):
    li = []
    for arg in args:
        li.append(np.nanmean(np.array(arg), axis=0))
    return li


def latest(*args):
    li = []
    for arg in args:
        li.append(arg[-1])
    return li


if __name__ == '__main__':
    ...
