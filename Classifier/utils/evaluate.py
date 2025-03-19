import torchmetrics.classification
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score, \
    hamming_loss, jaccard_score, \
    roc_curve, precision_recall_curve, auc, roc_auc_score, classification_report, confusion_matrix
import torch.nn.functional as F
import numpy as np


def compute_accuracy(preds, labels, num_classes):
    accuracy = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=num_classes
    )
    return accuracy(preds=preds.detach().cpu(), target=labels.detach().cpu())


def find_optimal_threshold(preds, labels, thresh, method=None):
    """
    :param preds: prediction of one class.
    :param labels: ground truth labels.
    :param method:  'None': threshold is default 0.5.
                    'ROC': Use Receiver Operating Characteristic (ROC) curve to find threshold.
                    'PRC': Use Precision-Recall curve to find threshold.
                    'Test': Use Pre set Thresholds.
    :return: optimal_threshold: the optimal threshold for the given class.
             processed_preds: prediction after thresholding.
             labels: processed labels.
    """

    labels = labels.cpu().squeeze().detach().numpy()
    preds_pos = preds[:, 1].cpu().detach().numpy()

    fpr, tpr, thresholds_roc = roc_curve(labels, preds_pos)
    roc_auc = auc(fpr, tpr)
    precisions, recalls, thresholds_prc = precision_recall_curve(labels, preds_pos)

    prc_auc = auc(recalls, precisions)

    if method == 'ROC':
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_roc[optimal_idx]
    elif method == 'PRC':
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_prc[optimal_idx]
    elif method == 'None':
        optimal_threshold = 0.5
    elif method == 'test':
        optimal_threshold = thresh

    processed_preds = (preds_pos > optimal_threshold).astype(int)

    return optimal_threshold, processed_preds, labels, roc_auc, prc_auc


def evaluate_multiclass_classifier(preds_dict, labels_dict, loss_dict, class_list):
    """
    :param preds_dict:
    :param labels_dict: label_name e.g., 'acl', 'meniscus tear','cartilage thickness loss'
    :param loss_dict:
    :param class_list: class_list e.g., 'high','low','not'
    :return: class_metrics
    """

    labels_combined = []
    preds_combined = []
    total_loss = 0
    class_losses = {}

    class_metrics = {}
    thresholds = {}

    for label_name in labels_dict:
        average_type = 'macro'
        prc_auc = 0.0
        thresholds[label_name] = 0.0
        labels = labels_dict[label_name].cpu().squeeze().detach().numpy()
        preds = preds_dict[label_name].cpu().detach().numpy()
        if np.unique(labels).size > 2:
            one_hot_labels = F.one_hot(labels_dict[label_name], num_classes=preds_dict[label_name].shape[1])
            roc_auc = roc_auc_score(one_hot_labels.cpu().squeeze().detach().numpy(), preds, multi_class='ovr')
        else:
            roc_auc = 0.0
        processed_preds = np.argmax(preds, axis=-1)

        labels_combined.append(labels)
        preds_combined.append(processed_preds)

        class_loss = loss_dict[label_name].mean().item()
        class_losses[label_name] = class_loss
        total_loss += class_loss
        # Generate classification report as a dictionary
        report = classification_report(labels, processed_preds, target_names=class_list, output_dict=True,
                                       zero_division=0)

        class_metrics[label_name] = {
            'Balanced_Accuracy': balanced_accuracy_score(labels, processed_preds),
            'Precision': precision_score(labels, processed_preds, zero_division=0, average=average_type),
            'Recall': recall_score(labels, processed_preds, zero_division=0, average=average_type),
            'F1 Score': f1_score(labels, processed_preds, zero_division=0, average=average_type),
            'Jaccard Similarity': jaccard_score(labels, processed_preds, zero_division=0, average=average_type),
            'ROC AUC': roc_auc,
            'PRC AUC': prc_auc,
            'Loss': class_loss,
            'metrics_per_class': report
        }

    return class_metrics


def evaluate_classifier(preds_dict, labels_dict, loss_dict, thresh, method=None):
    """
    :param preds_dict: prediction of one class.
    :param labels_dict: ground truth labels.
    :param method:  'None': threshold is default 0.5.
                    'ROC': Use Receiver Operating Characteristic (ROC) curve to find threshold.
                    'PRC': Use Precision-Recall curve to find threshold.
                    'Test': Use Pre set Thresholds.
    :return metrics_dict: A dictionary containing various multilabel metrics
    """
    labels_combined = []
    preds_combined = []
    total_loss = 0
    class_losses = {}

    class_metrics = {}
    thresholds = {}

    for class_name in labels_dict:
        average_type = 'binary'
        optimal_threshold, processed_preds, labels, roc_auc, prc_auc = find_optimal_threshold(
            preds_dict[class_name], labels_dict[class_name], thresh[class_name], method=method)
        thresholds[class_name] = optimal_threshold
        labels_combined.append(labels)
        preds_combined.append(processed_preds)
        class_loss = loss_dict[class_name].mean().item()
        class_losses[class_name] = class_loss
        total_loss += class_loss

        class_metrics[class_name] = {
            'Balanced_Accuracy': balanced_accuracy_score(labels, processed_preds),
            'Precision': precision_score(labels, processed_preds, zero_division=0, average=average_type),
            'Recall': recall_score(labels, processed_preds, zero_division=0, average=average_type),
            'F1 Score': f1_score(labels, processed_preds, zero_division=0, average=average_type),
            'Jaccard Similarity': jaccard_score(labels, processed_preds, zero_division=0, average=average_type),
            'ROC AUC': roc_auc,
            'PRC AUC': prc_auc,
            'Loss': class_loss
        }

    metrics_dict = {
        'Thresholds': thresholds,
        'Exact Match Ratio': 0.0,
        'Hamming Loss': 0.0,
        'Precision (Micro)': precision_score(labels_combined, preds_combined, average='micro', zero_division=0),
        'Precision (Macro)': precision_score(labels_combined, preds_combined, average='macro', zero_division=0),
        'Recall (Micro)': recall_score(labels_combined, preds_combined, average='micro', zero_division=0),
        'Recall (Macro)': recall_score(labels_combined, preds_combined, average='macro', zero_division=0),
        'F1 Score (Micro)': f1_score(labels_combined, preds_combined, average='micro', zero_division=0),
        'F1 Score (Macro)': f1_score(labels_combined, preds_combined, average='macro', zero_division=0),
        'Jaccard Similarity (Micro)': jaccard_score(labels_combined, preds_combined, average='micro', zero_division=0),
        'Jaccard Similarity (Macro)': jaccard_score(labels_combined, preds_combined, average='macro', zero_division=0),
        'Average Class Loss': total_loss / len(labels_dict),
        'Overall Loss': total_loss,
        'Class Metrics': class_metrics
    }

    return metrics_dict
