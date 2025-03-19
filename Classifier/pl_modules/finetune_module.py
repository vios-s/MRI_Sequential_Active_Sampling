import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import sys
import argparse
sys.path.append('../')
from models import MultiHeadResNet18, MultiHeadResNet50, KspaceNetRes50, MultiHeadSqueezeNet
from utils import compute_accuracy, evaluate_classifier

class FineTuneBinaryModule(LightningModule):
    def __init__(
            self,
            args: argparse.Namespace
    ):
        """_summary_


        """
        super().__init__()

        self.save_hyperparameters()
        self.model_type = args.model_type
        self.num_classes = args.num_classes
        self.lr = args.lr
        self.lr_backbone = args.lr_backbone
        self.input_type = args.log_name.split("_")[-1]
        self.lr_step_size = args.lr_step_size
        self.lr_gamma = args.lr_gamma
        self.weight_decay = args.weight_decay
        self.in_channel = args.in_channel
        self.label_names = args.label_names
        self.method = args.thresh_method
        self.thresh = args.thresh_dict
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.model =[]

        if args.model_type == 'resnet50':
            self.model = MultiHeadResNet50(args)
        elif args.model_type == 'resnet18':
            self.model = MultiHeadResNet18(args)
        elif args.model_type == 'kspacenet':
            self.model = KspaceNetRes50(args)
        elif args.model_type == 'squeezenet':
            self.model = MultiHeadSqueezeNet(args)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")


        # Load pretrained checkpoint
        if args.pretrained_model_checkpoint:
            checkpoint = torch.load(args.pretrained_model_checkpoint)
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                new_key = k.replace("model.", "")  # Remove the "model." prefix
                new_state_dict[new_key] = v
            del new_state_dict['criterion.weight']

            self.model.load_state_dict(new_state_dict)

            # Initialize head weights with small random values
            for name, param in self.model.named_parameters():
                if 'backbone' not in name:
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0.0, std=0.01)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        #Update loss weight later
        if args.loss_type in ["WCE", "focal_loss"]:
            self.loss_fn_weights = torch.tensor([1.1422, 1.0000])

        #   cart [0.2258, 1.000]
        #   acl [1.1422, 1.0000]


        if args.loss_type == "WCE":
            self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=self.loss_fn_weights.float().cuda())
        elif args.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

        print(self.model)


    def forward(self, input):

        return self.model(input)

    def training_step(self, batch, batch_idx):

        if self.input_type == 'kspace':
            input = batch.kspace
        elif self.input_type == 'recon':
            input = batch.recon
        elif self.input_type == 'image':
            input = batch.undersampled
        else:
            print(f'Invalid input type [{self.input_type}]')

        label = batch.label[self.label_names]
        output = self(input)


        if self.model_type == 'kspacenet':
            processed_output = F.log_softmax(output[0], dim=-1)
        else:
            processed_output = F.softmax(output[0], dim=-1)



        acc = compute_accuracy(
            preds=processed_output, labels=label.squeeze(1), num_classes=self.num_classes
        )

        loss = self.loss_fn(
            pred=processed_output, label=label.squeeze(1)
        )


        self.log(
            f"train/{self.label_names}_acc", acc, prog_bar=True, sync_dist=True
        )
        self.log(
            f"train/{self.label_names}_loss", loss.detach(), prog_bar=True, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        preds_dict = {}
        loss_dict = {}


        if self.input_type == 'kspace':
            input = batch.kspace
        elif self.input_type == 'recon':
            input = batch.recon
        elif self.input_type == 'image':
            input = batch.undersampled
        else:
            print(f'Invalid input type [{self.input_type}]')

        label = batch.label
        output = self(input)

        if self.model_type == 'kspacenet':
            processed_output = F.log_softmax(output[0], dim=-1)
        else:
            processed_output = F.softmax(output[0], dim=-1)

        loss = self.loss_fn(
            pred=processed_output, label=label[self.label_names].squeeze(1)
        )
        loss_dict[self.label_names] = loss.detach()
        preds_dict[self.label_names] = processed_output


        output_log = {
                "predictions": preds_dict,
                "labels": batch.label,
                "loss_dict": loss_dict
                 }

        self.validation_step_outputs.append(output_log)

        return output_log

    def test_step(self, batch, batch_idx):
        preds_dict = {}
        loss_dict = {}

        if self.input_type == 'kspace':
            input = batch.kspace
        elif self.input_type == 'recon':
            input = batch.recon
        elif self.input_type == 'image':
            input = batch.undersampled
        else:
            print(f'Invalid input type [{self.input_type}]')

        label = batch.label
        output = self(input)

        if self.model_type == 'kspacenet':
            processed_output = F.log_softmax(output[0], dim=-1)
        else:
            processed_output = F.softmax(output[0], dim=-1)

        loss = self.loss_fn(
            pred=processed_output, label=label[self.label_names].squeeze(1)
        )

        loss_dict[self.label_names] = loss.detach()
        preds_dict[self.label_names] = processed_output

        output_log = {
            "predictions": preds_dict,
            "labels": batch.label,
            "loss_dict": loss_dict
        }

        self.test_step_outputs.append(output_log)
        return output_log

    def loss_fn(
        self, pred: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        batch_size = pred.shape[0]
        label = label

        assert label.shape == (batch_size, )
        return self.criterion(pred, label)

    def configure_optimizers(self):
        # Separate parameter groups for backbone and classifier
        backbone_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.lr_backbone},
            {'params': classifier_params, 'lr': self.lr}
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        combined_preds, combined_labels ,combined_losses= self.aggregate_logs(self.validation_step_outputs)
        metrics_dict = evaluate_classifier(combined_preds, combined_labels, combined_losses, self.thresh, method=self.method)
        for class_name in combined_labels:
            class_metrics = metrics_dict['Class Metrics'][class_name]
            self.log(f"val_{class_name}/thresholds", metrics_dict['Thresholds'][class_name], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/balanced_accuracy", class_metrics['Balanced_Accuracy'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/precision", class_metrics['Precision'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/recall", class_metrics['Recall'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/f1_score", class_metrics['F1 Score'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/jaccard_similarity", class_metrics['Jaccard Similarity'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/roc_auc", class_metrics['ROC AUC'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/prc_auc", class_metrics['PRC AUC'], prog_bar=True, sync_dist=True)
            self.log(f"val_{class_name}/loss", class_metrics['Loss'], prog_bar=True, sync_dist=True)

        # Log overall metrics
        self.log("val_exact_match_ratio", metrics_dict['Exact Match Ratio'], prog_bar=True, sync_dist=True)
        self.log("val_hamming_loss", metrics_dict['Hamming Loss'], prog_bar=True, sync_dist=True)

        # Log micro-averaged metrics
        self.log("val_micro/precision", metrics_dict['Precision (Micro)'], prog_bar=True, sync_dist=True)
        self.log("val_micro/recall", metrics_dict['Recall (Micro)'], prog_bar=True, sync_dist=True)
        self.log("val_micro/f1_score", metrics_dict['F1 Score (Micro)'], prog_bar=True, sync_dist=True)
        self.log("val_micro/jaccard_similarity", metrics_dict['Jaccard Similarity (Micro)'], prog_bar=True, sync_dist=True)

        # Log macro-averaged metrics
        self.log("val_macro/precision", metrics_dict['Precision (Macro)'], prog_bar=True, sync_dist=True)
        self.log("val_macro/recall", metrics_dict['Recall (Macro)'], prog_bar=True, sync_dist=True)
        self.log("val_macro/f1_score", metrics_dict['F1 Score (Macro)'], prog_bar=True, sync_dist=True)
        self.log("val_macro/jaccard_similarity", metrics_dict['Jaccard Similarity (Macro)'], prog_bar=True, sync_dist=True)

        # Log losses
        self.log("val_average_class_loss", metrics_dict['Average Class Loss'], prog_bar=True, sync_dist=True)
        self.log("val_overall_loss", metrics_dict['Overall Loss'], prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        combined_preds, combined_labels, combined_losses= self.aggregate_logs(self.test_step_outputs)
        metrics_dict = evaluate_classifier(combined_preds, combined_labels, combined_losses, self.thresh, method=self.method)
        for class_name in combined_labels:
            class_metrics = metrics_dict['Class Metrics'][class_name]
            self.log(f"test_{class_name}/thresholds", metrics_dict['Thresholds'][class_name], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/balanced_accuracy", class_metrics['Balanced_Accuracy'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/precision", class_metrics['Precision'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/recall", class_metrics['Recall'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/f1_score", class_metrics['F1 Score'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/jaccard_similarity", class_metrics['Jaccard Similarity'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/roc_auc", class_metrics['ROC AUC'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/prc_auc", class_metrics['PRC AUC'], prog_bar=True, sync_dist=True)
            self.log(f"test_{class_name}/loss", class_metrics['Loss'], prog_bar=True, sync_dist=True)

        # Log overall metrics
        self.log("test_exact_match_ratio", metrics_dict['Exact Match Ratio'], prog_bar=True, sync_dist=True)
        self.log("test_hamming_loss", metrics_dict['Hamming Loss'], prog_bar=True, sync_dist=True)

        # Log micro-averaged metrics
        self.log("test_micro/precision", metrics_dict['Precision (Micro)'], prog_bar=True, sync_dist=True)
        self.log("test_micro/recall", metrics_dict['Recall (Micro)'], prog_bar=True, sync_dist=True)
        self.log("test_micro/f1_score", metrics_dict['F1 Score (Micro)'], prog_bar=True, sync_dist=True)
        self.log("test_micro/jaccard_similarity", metrics_dict['Jaccard Similarity (Micro)'], prog_bar=True, sync_dist=True)

        # Log macro-averaged metrics
        self.log("test_macro/precision", metrics_dict['Precision (Macro)'], prog_bar=True, sync_dist=True)
        self.log("test_macro/recall", metrics_dict['Recall (Macro)'], prog_bar=True, sync_dist=True)
        self.log("test_macro/f1_score", metrics_dict['F1 Score (Macro)'], prog_bar=True, sync_dist=True)
        self.log("test_macro/jaccard_similarity", metrics_dict['Jaccard Similarity (Macro)'], prog_bar=True, sync_dist=True)

        # Log losses
        self.log("test_average_class_loss", metrics_dict['Average Class Loss'], prog_bar=True, sync_dist=True)
        self.log("test_overall_loss", metrics_dict['Overall Loss'], prog_bar=True, sync_dist=True)


        self.test_step_outputs.clear()

    @staticmethod
    def aggregate_logs(val_logs):
        combined_preds = {}
        combined_labels = {}
        combined_losses = {}

        for val_log in val_logs:
            for key in val_log['predictions']:
                if key not in combined_preds:
                    combined_preds[key] = []
                    combined_labels[key] = []
                    combined_losses[key] = []
                combined_preds[key].append(val_log['predictions'][key])
                combined_labels[key].append(val_log['labels'][key])
                combined_losses[key].append(val_log['loss_dict'][key])

        for key in combined_preds:
            combined_preds[key] = torch.cat(combined_preds[key], dim=0)
            combined_labels[key] = torch.cat(combined_labels[key], dim=0)
            combined_losses[key] = torch.stack(combined_losses[key], dim=0)

        return combined_preds, combined_labels, combined_losses
