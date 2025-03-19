import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import sys
import argparse
sys.path.append('../')
from models import MultiHeadResNet18, MultiHeadResNet50, KspaceNetRes50, MultiHeadSqueezeNet


class Binary_ClassificationModule(LightningModule):
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

        #Update loss weight later
        if args.loss_type in ["WCE", "focal_loss"]:
                self.loss_fn_weights = torch.tensor([0.04, 0.96])


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
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )
        return [optim], [scheduler]


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
