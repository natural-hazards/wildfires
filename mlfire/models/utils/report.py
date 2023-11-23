
from mlfire.utils.functool import lazy_import

# lazy imports
_np = lazy_import('numpy')


def classification_report(labels_true: _np.ndarray, labels_pred: _np.ndarray) -> None:

    imblearn_metrics = lazy_import('imblearn.metrics')
    sklearn_metrics = lazy_import('sklearn.metrics')

    # print classification report and its imbalanced version
    print('Classification report:\n')
    print(sklearn_metrics.classification_report(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))

    print('\nClassification report (imbalanced):\n')
    print(imblearn_metrics.classification_report_imbalanced(
        labels_true[~_np.isnan(labels_true)], labels_pred[~_np.isnan(labels_true)]
    ))

    """
    computing IoU (Intersection over Union) for positive samples
    """

    iou_score_p = sklearn_metrics.jaccard_score(y_true=labels_true[~_np.isnan(labels_true)],
                                                y_pred=labels_pred[~_np.isnan(labels_true)])
    """
    computing IoU  for negative samples
    """

    labels_true = labels_true[~_np.isnan(labels_true)]
    labels_pred = labels_pred[~_np.isnan(labels_pred)]

    val_min = labels_true.min(); val_max = labels_true.max()

    if min != 0:
        labels_true[labels_true == val_min] = 0
        labels_pred[labels_pred == val_min] = 0
    if max != 1:
        labels_true[labels_true == val_max] = 1
        labels_pred[labels_pred == val_max] = 1

    labels_true = labels_true.astype(_np.int32)
    labels_true = _np.where((labels_true == 0) | (labels_true == 1), labels_true ^ 1, labels_true)

    labels_pred = labels_pred.astype(_np.int32)
    labels_pred = _np.where((labels_pred == 0) | (labels_pred == 1), labels_pred ^ 1, labels_pred)

    iou_score_n = sklearn_metrics.jaccard_score(y_true=labels_true, y_pred=labels_pred)

    # computing mean intersection over union
    iou_score_m = (iou_score_p + iou_score_n) / 2.

    print('\nIoU- (intersection over union): {:.2f}'.format(iou_score_n))
    print('\nIoU+ (intersection over union): {:.2f}'.format(iou_score_p))
    print('\nmIoU (mean IoU): {:.2f}'.format(iou_score_m))


def plot_aucroc(labels_true: _np.ndarray, labels_pred: _np.ndarray, ax=None) -> None:

    roc = lazy_import('mlfire.models.utils.roc')

    auc_roc = roc.AucRoc(labels_true=labels_true, labels_pred=labels_pred)
    auc_roc.plot(ax=ax)


def plot_cmat(labels_true: _np.ndarray, labels_pred: _np.ndarray, ax=None) -> None:

    cmat = lazy_import('mlfire.models.utils.cmat')

    cm = cmat.ConfusionMatrix(
        y_true=labels_true[~_np.isnan(labels_true)],
        y_pred=labels_pred[~_np.isnan(labels_true)],
        labels=['Background', 'Fire']
    )

    cm.plot(ax=ax)


def show_report(labels_true: _np.ndarray, labels_pred: _np.ndarray, with_aucroc: bool = False, with_cmat: bool = False,
                with_scores: bool = False) -> None:

    if with_scores: classification_report(labels_true=labels_true, labels_pred=labels_pred)

    if with_cmat and with_aucroc:
        if with_scores: print('\n')

        plt_pylab = lazy_import('matplotlib.pylab')
        _, axes = plt_pylab.subplots(1, 2, figsize=(10, 5))

        plot_aucroc(labels_true=labels_true, labels_pred=labels_pred, ax=axes[0])
        plot_cmat(labels_true=labels_true, labels_pred=labels_pred, ax=axes[1])
    elif with_cmat:
        plot_cmat(labels_true=labels_true, labels_pred=labels_pred)
    elif with_aucroc:
        plot_aucroc(labels_true=labels_true, labels_pred=labels_pred)
