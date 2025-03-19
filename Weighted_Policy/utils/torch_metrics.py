import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cross_entropy(outputs, label):
    loss_fn_weights = torch.tensor([0.5, 0.5])
    criterion = nn.CrossEntropyLoss(reduction='mean', weight=loss_fn_weights.float().cuda())
    cross_entropy = criterion(outputs, label)
    # Using F.cross_entropy directly, without one-hot encoding label

    return cross_entropy.item()


def compute_batch_metrics(processed_preds, labels, average_type='binary'):

    metrics_dict = {
        'Balanced_Accuracy': balanced_accuracy_score(labels, processed_preds),
        'Precision': precision_score(labels, processed_preds, zero_division=0, average=average_type),
        'Recall': recall_score(labels, processed_preds, zero_division=0, average=average_type),
        'F1 Score': f1_score(labels, processed_preds, zero_division=0, average=average_type)
    }

    return metrics_dict

def calculate_classification_metrics(confusion_matrix):
    # Sum the confusion matrix across all steps in the epoch
    sum_confusion_matrix = torch.sum(torch.stack(confusion_matrix), dim=0)

    # Calculate metrics
    true_positive = sum_confusion_matrix[1, 1].item()
    false_positive = sum_confusion_matrix[0, 1].item()
    false_negative = sum_confusion_matrix[1, 0].item()
    true_negative = sum_confusion_matrix[0, 0].item()

    # Use torchmetrics to calculate metrics
    # auroc_metric = AUROC()

    # Calculate metrics
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # auc = auroc_metric(outputs[:, 1], label)  # Assuming outputs is a tensor with probability scores for the positive class

    metrics_dict = {
        'recall': recall.item(),
        'specificity': specificity.item(),
        'precision': precision,
        'f1_score': f1_score,
        # 'auc': auc.item(),
    }

    return metrics_dict

