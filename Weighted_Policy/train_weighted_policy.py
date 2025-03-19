import logging
import time
import click
import yaml
import argparse
import pathlib
import wandb
from pathlib import Path
import pytorch_lightning as pl
import os
import torch
import numpy as np
import csv
from tensorboardX import SummaryWriter
import h5py
from utils.torch_metrics import compute_cross_entropy, compute_batch_metrics
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve
from utils.utils import (save_json, build_optim, count_parameters,
                         count_trainable_parameters, count_untrainable_parameters)
from data.data_loading import create_data_loader
from inference_model.inference_model_utils import load_infer_model

from policy_model.policy_model_utils import (build_policy_model, load_policy_model, save_policy_model,
                                             compute_backprop_trajectory,
                                             compute_next_step_inference, get_policy_probs)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(args, epoch, reward_model_list, model, loader, optimiser, writer):

    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(loader)

    cbatch = 0  # Counter for spreading single backprop batch over multiple data loader batches
    for it, data in enumerate(loader):  # Loop over data points
        cbatch += 1
        masked_kspace, zf, mask, kspace, gt, fname, slice, label, max_value = data
        label = torch.stack(list(label.values())).to(args.device).squeeze(-1)
        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = torch.view_as_real(kspace).to(args.device)
        masked_kspace = torch.view_as_real(masked_kspace).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)
        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.to(args.device)
        gt = gt.to(args.device)
        # Base inference model forward pass
        diseased_feature_map, diseased_outputs = reward_model_list[0](zf)
        severity_feature_map, severity_outputs = reward_model_list[1](zf)
        image_input = torch.cat((diseased_feature_map, severity_feature_map), dim=1)
        outputs = [diseased_outputs, severity_outputs]

        if cbatch == 1:  # Only after backprop is performed
            optimiser.zero_grad()

        action_list = []
        logprob_list = []
        reward_list = []

        for step in range(args.acquisition_steps):  # Loop over acquisition steps

            loss, mask, masked_kspace, image_input, outputs = compute_backprop_trajectory(args, kspace, masked_kspace,
                                                                                          mask, outputs, image_input,
                                                                                          label, model,
                                                                                          reward_model_list, step,
                                                                                          action_list, logprob_list,
                                                                                          reward_list)

            # Loss logging
            epoch_loss[step] += loss.item() / len(loader) * gt.size(0) / args.batch_size
            report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

        # Backprop if we've reached the prerequisite number of dataloader batches
        if cbatch == args.batches_step:
            optimiser.step()
            cbatch = 0

        # Logging: note that loss values mean little, as the Policy Gradient loss is not a true loss.
        if it % args.report_interval == 0:
            if it == 0:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e3)
                                      for i, l in enumerate(report_loss)])
            else:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e3) for i, l in enumerate(report_loss)])
            logging.info(
                f'Epoch = [{epoch:3d}/{args.max_epochs:3d}], '
                f'Iter = [{it:4d}/{len(loader):4d}], '
                f'Time = {time.perf_counter() - start_iter:.2f}s, '
                f'Avg Loss per step x1e3 = [{loss_str}] ',
            )
            report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    return np.mean(epoch_loss), time.perf_counter() - start_epoch


def evaluate(args, epoch, reward_model_list, model, loader, writer, partition):

    model.eval()
    balanced_accuracy = [[], [], []]
    recall = [[], [], []]
    f1 = [[], [], []]
    auc = [[], [], []]
    tbs = 0  # data set size counter
    start = time.perf_counter()
    all_diseased_outputs = [[] for _ in range(args.acquisition_steps + 1)]
    all_severity_outputs = [[] for _ in range(args.acquisition_steps + 1)]
    all_diseased_label = [[] for _ in range(args.acquisition_steps + 1)]
    all_task_label = [[] for _ in range(args.acquisition_steps + 1)]

    for it, data in enumerate(loader):
        masked_kspace, zf, mask, kspace, gt, fname, slice, label, max_value = data
        label = torch.stack(list(label.values())).to(args.device).squeeze(-1)
        label[1] = torch.where(label[0] == 0, torch.tensor(2), label[1]).int()

        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = torch.view_as_real(kspace).to(args.device)
        masked_kspace = torch.view_as_real(masked_kspace).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)

        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.to(args.device)

        tbs += mask.size(0)

        # Base inference model forward pass
        diseased_feature_map, diseased_outputs = reward_model_list[0](zf)
        severity_feature_map, severity_outputs = reward_model_list[1](zf)
        image_input = torch.cat((diseased_feature_map, severity_feature_map), dim=1)

        # print('initial computing')
        all_diseased_outputs[0].extend(diseased_outputs.detach().cpu().numpy())
        all_severity_outputs[0].extend(severity_outputs.detach().cpu().numpy())
        all_diseased_label[0].extend(label[0].cpu())
        all_task_label[0].extend(label[1].cpu())

        if args.save_mask:

            os.makedirs(f'./mask_{args.log_name}/', exist_ok=True)
            mask_all = mask.squeeze().unsqueeze(0)

        for step in range(args.acquisition_steps):
            policy, probs = get_policy_probs(model, image_input, mask)
            if step == 0:
                actions = torch.multinomial(probs.squeeze(1), args.test_trajectories, replacement=True)
            else:
                actions = policy.sample()

            # Samples trajectories in parallel
            # For evaluation we can treat greedy and non-greedy the same: in both cases we just simulate
            # num_test_trajectories acquisition trajectories in parallel for each slice in the batch, and store
            # the average cross entropy score every time step.
            mask, _, _, image_input, outputs = compute_next_step_inference(reward_model_list, kspace, masked_kspace, mask,
                                                                 actions, args.recon_size)

            if args.save_mask:
                mask_all = torch.cat([mask_all, mask.squeeze().unsqueeze(0)])

            all_diseased_outputs[step + 1].extend(outputs[0].squeeze(1).detach().cpu().numpy())
            all_severity_outputs[step + 1].extend(outputs[1].squeeze(1).detach().cpu().numpy())
            all_diseased_label[step+1].extend(label[0].cpu())
            all_task_label[step+1].extend(label[1].cpu())

        if args.save_mask:
            mask_all_numpy = mask_all.cpu().numpy()  # Add .cpu() if tensor is on GPU
            np.save(f'./mask_{args.log_name}/mask_{it}.npy', mask_all_numpy)

    for num in range(args.acquisition_steps + 1):
        step_diseased_label = np.array(all_diseased_label[num])
        step_task_label = np.array(all_task_label[num])

        step_diseased_outputs = np.array(all_diseased_outputs[num])[:, 1]
        step_task_outputs = np.array(all_severity_outputs[num])[:, 1]

        step_diseased_preds = (step_diseased_outputs > args.thresh[0]).astype(int)
        step_severity_preds = (step_task_outputs > args.thresh[1]).astype(int)
        step_task_preds = np.where(step_diseased_preds == 0, np.array(2), step_severity_preds)

        if (step_task_label == 2).all():
            print("All samples belong to the '-1' class. Skipping evaluation for this task.")
        else:
            # Exclude -1 labels
            valid_indices = step_task_label != 2
            valid_step_severity_label = step_task_label[valid_indices]
            valid_step_severity_outputs = step_task_outputs[valid_indices]
            valid_step_severity_preds = step_severity_preds[valid_indices]

        # single task evaluation for diseased
        diseased_fpr, diseased_tpr, _ = roc_curve(step_diseased_label, step_diseased_outputs)
        step_diseased_auc = sk_auc(diseased_fpr, diseased_tpr)
        diseased_metrics = compute_batch_metrics(step_diseased_preds, step_diseased_label)

        # single task evaluation for severity
        severity_fpr, severity_tpr, _ = roc_curve(valid_step_severity_label, valid_step_severity_outputs)
        step_severity_auc = sk_auc(severity_fpr, severity_tpr)
        severity_metrics = compute_batch_metrics(valid_step_severity_preds, valid_step_severity_label)

        # multi-objective evaluation
        task_metrics = compute_batch_metrics(step_task_preds, step_task_label, average_type='macro')

        # Store metrics for each task
        auc[0].append(step_diseased_auc)
        auc[1].append(step_severity_auc)
        balanced_accuracy[0].append(diseased_metrics['Balanced_Accuracy'])
        balanced_accuracy[1].append(severity_metrics['Balanced_Accuracy'])
        recall[0].append(diseased_metrics['Recall'])
        recall[1].append(severity_metrics['Recall'])
        f1[0].append(diseased_metrics['F1 Score'])
        f1[1].append(severity_metrics['F1 Score'])

        # Compute combined (multi-objective) metrics
        auc[2].append((step_severity_auc + step_diseased_auc) / 2)
        balanced_accuracy[2].append(task_metrics['Balanced_Accuracy'])
        recall[2].append(task_metrics['Recall'])
        f1[2].append(task_metrics['F1 Score'])

    # Logging
    if partition in ['Val', 'Train', 'Test']:
        for step, val in enumerate(recall):
            writer.add_scalar(f'{partition}_balanced_accuracy_step{step}', balanced_accuracy[2][step], epoch)
            writer.add_scalar(f'{partition}_recall_step{step}', recall[2][step], epoch)
            writer.add_scalar(f'{partition}_F1_step{step}', f1[2][step], epoch)
            writer.add_scalar(f'{partition}_AUC_step{step}', auc[2][step], epoch)
    else:
        raise ValueError(f"'partition' should be in ['Train', 'Val', 'Test'], not: {partition}")

    return balanced_accuracy, recall, f1, auc, time.perf_counter() - start


def train_and_eval(args, reward_model_list):
    # Improvement model to train
    model = build_policy_model(args)
    optimiser = build_optim(args, model.parameters())
    start_epoch = 0
    # Create directory to store results in
    savestr = Path(args.log_name)
    args.run_dir = Path(args.log_path) / Path(args.log_name) / savestr
    os.makedirs(args.run_dir, exist_ok=True)

    # Logging
    logging.info(reward_model_list[0])
    logging.info(reward_model_list[1])
    logging.info(model)
    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    # Parameter counting
    logging.info('Disease Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(reward_model_list[0]), count_trainable_parameters(reward_model_list[0]),
        count_untrainable_parameters(reward_model_list[0])))
    logging.info('Severity Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(reward_model_list[1]), count_trainable_parameters(reward_model_list[1]),
        count_untrainable_parameters(reward_model_list[1])))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    # Create data loaders
    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val', shuffle=False)

    if not args.resume_train:
        do_and_log_evaluation(args, -1, reward_model_list, model, dev_loader, writer, 'Val')

    for epoch in range(start_epoch, args.max_epochs):
        train_loss, train_time = train_epoch(args, epoch, reward_model_list, model, train_loader, optimiser, writer)
        logging.info(
            f'Epoch = [{epoch + 1:3d}/{args.max_epochs:3d}] TrainLoss = {train_loss:.3g} TrainTime = {train_time:.2f}s '
        )
        do_and_log_evaluation(args, epoch, reward_model_list, model, dev_loader, writer, 'Val')

        scheduler.step()
        save_policy_model(args, args.run_dir, epoch, model, optimiser)
    writer.close()


def do_and_log_evaluation(args, epoch, infer_model, model, loader, writer, partition):
    """
    Helper function for logging.
    """
    balanced_accuracy, recall, f1, AUC, score_time = evaluate(args, epoch, infer_model, model, loader,
                                                                             writer, partition)
    for idx, task_name in enumerate(['diseased', 'severity', 'sequential']):
        balanced_accuracy_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(balanced_accuracy[idx])])
        recall_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(recall[idx])])
        f1_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(f1[idx])])
        AUC_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(AUC[idx])])
        logging.info(f'Results for {task_name} diagnosis:')
        logging.info(f'{partition}_Balanced_Accuracy = [{balanced_accuracy_str}]')
        logging.info(f'{partition}_Recall = [{recall_str}]')
        logging.info(f'{partition}_F1 Score = [{f1_str}]')
        logging.info(f'{partition}_AUC = [{AUC_str}]')
        logging.info(f'{partition}_ScoreTime = {score_time:.2f}s')

    if partition == 'Test':
        # Create CSV filename with timestamp
        csv_file = f'{args.log_name}_{partition}_seed{args.seed}.csv'
        # Write results to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Create header row: [Task, Metric, Name] + [Step_0, Step_1, ..., Step_80]
            header = ['Task', 'Metric', 'Name'] + [f'Step_{i}' for i in range(args.acquisition_steps+1)]
            writer.writerow(header)

            # For each task and its metrics
            for idx, task in enumerate(['diseased', 'severity', 'sequential']):
                metrics_dict = {
                    'Balanced_Accuracy': balanced_accuracy[idx],
                    'Recall': recall[idx],
                    'F1': f1[idx],
                    'AUC': AUC[idx]
                }

                # Write a row for each metric with all 80 steps
                for metric_name, values in metrics_dict.items():
                    # Create row: [task, metric_name, ''] + [value_0, value_1, ..., value_79]
                    row = [task, metric_name, '']  # Empty string for Name column
                    row.extend([f'{v:.3f}' for v in values])
                    writer.writerow(row)

        logging.info(f'Results saved to {csv_file} (score_time: {score_time:.2f}s)')

def test(args, reward_model_list):
    """
    Performs evaluation of a pre-trained policy model.

    :param args: Argument object containing evaluation parameters.
    :param recon_model: reconstruction model.
    """
    model, policy_args = load_policy_model(pathlib.Path(args.ckpt_path))
    # Overwrite number of trajectories to test on
    policy_args.test_trajectories = args.test_trajectories
    policy_args.center_fractions = args.center_fractions
    policy_args.acquisition_steps = args.acquisition_steps
    if args.data_path is not None:  # Overwrite data path if provided
        policy_args.data_path = args.data_path

    # Logging of policy model
    logging.info(args)
    logging.info(reward_model_list[0])
    logging.info(reward_model_list[1])
    logging.info(model)
    if args.wandb:
        wandb.config.update(args)
        wandb.watch(model, log='all')
    # Initialise summary writer
    writer = SummaryWriter(log_dir=policy_args.run_dir / 'summary')

    # Parameter counting

    logging.info('Disease Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(reward_model_list[0]), count_trainable_parameters(reward_model_list[0]),
        count_untrainable_parameters(reward_model_list[0])))
    logging.info('Severity Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(reward_model_list[1]), count_trainable_parameters(reward_model_list[1]),
        count_untrainable_parameters(reward_model_list[1])))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))
    policy_args = args
    # Create data loader
    test_loader = create_data_loader(policy_args, 'test', shuffle=False)
    # test_data_range_dict = create_data_range_dict(policy_args, test_loader)

    do_and_log_evaluation(policy_args, -1, reward_model_list, model, test_loader, writer, 'Test')

    writer.close()


def main(args):
    """
    Wrapper for training and testing of policy models.
    """
    logging.info(args)
    diseased_infer_model = load_infer_model(args, task='diseased')
    severity_infer_model = load_infer_model(args, task='severity')
    diseased_infer_model = diseased_infer_model.to(args.device)
    severity_infer_model = severity_infer_model.to(args.device)
    ##load classificatio model instead and save classification parameters
    reward_model_list = [diseased_infer_model, severity_infer_model]
    # Policy model to train
    if args.mode == 'train':
        train_and_eval(args, reward_model_list)
    else:
        test(args, reward_model_list)


def build_args(config_path):
    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)

    current_time = time.strftime("%m-%d_%H")
    args.default_root_dir = Path(args.log_path) / args.log_name / current_time
    args.run_name = current_time

    args.label_names = [args.diseased_label_name, args.severity_label_name]


    if not args.default_root_dir.exists():
        args.default_root_dir.mkdir(parents=True)

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    return args


@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option("--seed", required=True, type=int, help="Random Seed.")
def wrap_main(config_path, seed):
    """
    Wrapper for the entire script. Performs some setup, such as setting seed and starting wandb.
    """
    args = build_args(config_path)
    args.seed = seed
    if args.seed != 0:
        pl.seed_everything(args.seed)

    args.milestones = args.milestones + [0, args.max_epochs - 1]

    main(args)


if __name__ == '__main__':
    wrap_main()
