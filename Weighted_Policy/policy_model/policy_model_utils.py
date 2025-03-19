import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from .policy_model_def import build_policy_model

sys.path.append('..')
from utils.utils import build_optim
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, normalize, normalize_instance
from utils.torch_metrics import compute_cross_entropy, compute_batch_metrics


def save_policy_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=exp_dir / 'latest_model.pt'
    )

    if epoch in args.milestones:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=exp_dir / f'model_{epoch}.pt'
        )


def load_policy_model(checkpoint_file, optim=False):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_policy_model(args)

    if not optim:
        # No gradients for this model
        for param in model.parameters():
            param.requires_grad = False

    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return model, args, start_epoch, optimizer

    del checkpoint
    return model, args


def get_new_zf(masked_kspace, recon_size):
    masked_kspace = masked_kspace.squeeze(1)

    image_batch = torch.zeros([masked_kspace.size(0), masked_kspace.size(3), recon_size[0], recon_size[1]])
    for num in range(masked_kspace.size(0)):
        single_zf = complex_center_crop(ifft2c(masked_kspace[num, ...]), recon_size)
        single_zf = single_zf.permute(2, 0, 1)
        single_zf, means, stds = normalize_instance(single_zf, eps=0.0)
        image_batch[num, ...] = single_zf
    return image_batch.squeeze(1), means, stds


def acquire_rows_in_batch_parallel(k, mk, mask, to_acquire):
    if mask.size(1) == mk.size(1) == to_acquire.size(1):
        m_exp = mask
        mk_exp = mk
    else:
        # We have to initialise trajectories: every row in to_acquire corresponds to a trajectory.
        m_exp = mask.repeat(1, to_acquire.size(1), 1, 1, 1)
        mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
    # Loop over slices in batch
    for sl, rows in enumerate(to_acquire):
        # Loop over indices to acquire
        for index, row in enumerate(rows):  # Will only be single index if first case (see comment above)
            m_exp[sl, index, :, row.item(), :] = 1.
            mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
    return m_exp, mk_exp


def compute_next_step_inference(infer_model, kspace, masked_kspace, mask, next_rows, recon_size):

    image_input = []
    diseased_outputs = []
    severity_outputs = []
    zf = []
    mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, next_rows)
    channel_size = masked_kspace.shape[1]
    pha_res = masked_kspace.size(-3)
    fre_res = masked_kspace.size(-2)
    # Combine batch and channel dimension for parallel computation if necessary
    masked_kspace = masked_kspace.view(mask.size(0), channel_size, pha_res, fre_res, 2)
    for num in range(channel_size):
        single_mp = masked_kspace[:, num].unsqueeze(1)
        single_zf, _, _ = get_new_zf(single_mp, recon_size)
        # Base inference model forward pass
        single_zf = single_zf.to('cuda')
        diseased_feature_map, single_diseased_outputs = infer_model[0](single_zf)
        severity_feature_map, single_severity_outputs = infer_model[1](single_zf)
        single_feature_map = torch.cat((diseased_feature_map, severity_feature_map), dim=1)
        single_image_input = single_feature_map

        image_input.append(single_image_input)
        diseased_outputs.append(single_diseased_outputs)
        severity_outputs.append(single_severity_outputs)
        zf.append(single_zf)


    # Reshape back to B X C (=parallel acquisitions) x H x W
    outputs = [torch.stack(diseased_outputs, dim=1),
               torch.stack(severity_outputs, dim=1)]

    stacked_image_input = torch.stack(image_input, dim=1)
    zf = torch.stack(zf, dim=1)
    final_image_input = stacked_image_input.view(mask.size(0), channel_size, 2, single_image_input.size(-2),
                                                 single_image_input.size(-1))
    zf = zf.view(mask.size(0), channel_size, 2, recon_size[0], recon_size[1])
    masked_kspace = masked_kspace.view(mask.size(0), channel_size, pha_res, fre_res, 2)
    return mask, masked_kspace, zf, final_image_input, outputs


def get_policy_probs(model, input_image, mask):
    channel_size = mask.shape[1]
    res = mask.size(-2)
    # Reshape trajectory dimension into batch dimension for parallel forward pass
    input_image = input_image.view(mask.size(0) * channel_size, 2, input_image.size(-2), input_image.size(-1))
    # Obtain policy model logits
    output = model(input_image)
    # no previous mask as input
    # Reshape trajectories back into their own dimension
    output = output.view(mask.size(0), channel_size, res)  # [batch,1,res]

    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0).squeeze(-1).squeeze(-2).float()
    logits = torch.where(loss_mask.bool(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs


def compute_scores(args, outputs, label):
    cross_entropy = compute_cross_entropy(outputs, label)
    metrics = compute_batch_metrics(outputs, label)
    return cross_entropy, metrics


def compute_backprop_trajectory(args, kspace, masked_kspace, mask, outputs, image_input, label, model, infer_model,
                                step, action_list, logprob_list, reward_list):
    diseased_cross_entropy_scores = []
    severity_cross_entropy_scores = []
    loss_fn_weights = torch.tensor([1.0374, 27.7213])
    diseased_criteria = nn.BCELoss(reduction='none')
    severity_criteria = nn.BCELoss(reduction='none', weight=loss_fn_weights.float().cuda())

    # Base score from which to calculate acquisition rewards
    diseased_base_scores = diseased_criteria(outputs[0], F.one_hot(label[0], outputs[0].shape[-1]).float()).mean(dim=-1)
    severity_base_scores = severity_criteria(outputs[1], F.one_hot(label[1], outputs[1].shape[-1]).float()).mean(dim=-1)

    # Get policy and probabilities.
    policy, probs = get_policy_probs(model, image_input, mask)

    if step == 0 or args.greedy_action:  # probs has shape batch x 1 x res
        actions = torch.multinomial(probs.squeeze(1), args.num_trajectories, replacement=True)
        actions = actions.unsqueeze(1)  # batch x num_traj -> batch x 1 x num_traj
        # probs shape = batch x 1 x resfargs
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(1)
        actions = actions.squeeze(1)

    # Obtain rewards in parallel by taking actions in parallel
    mask, masked_kspace, zf, image_input, outputs = compute_next_step_inference(infer_model, kspace,
                                                                                masked_kspace, mask, actions,
                                                                                args.recon_size)
    for ch in range(masked_kspace.shape[1]):
        single_diseased_cross_entropy_scores = diseased_criteria(outputs[0][:, ch, :],
                                                                 F.one_hot(label[0], outputs[0].shape[-1]).float())
        diseased_cross_entropy_scores.append(single_diseased_cross_entropy_scores.mean(dim=-1))
        single_severity_cross_entropy_scores = severity_criteria(outputs[1][:, ch, :],
                                                                 F.one_hot(label[1], outputs[1].shape[-1]).float())
        severity_cross_entropy_scores.append(single_severity_cross_entropy_scores.mean(dim=-1))

    # batch x num_trajectories
    diseased_action_rewards = diseased_base_scores.unsqueeze(-1) - torch.stack(diseased_cross_entropy_scores,
                                                                               dim=0).transpose(1, 0)
    severity_action_rewards = severity_base_scores.unsqueeze(-1) - torch.stack(severity_cross_entropy_scores,
                                                                               dim=0).transpose(1, 0)
    if (label[0] == 0).all():
        print("All samples belong to the '0' class. Skipping evaluation for this task.")
        # Exclude -1 labels

    valid_indices = label[0] != 0
    severity_action_rewards[~valid_indices] = 0

    severity_weight = 0.5 * (1 - np.cos(np.pi * (step + 1) / args.acquisition_steps + np.pi * args.weight_beta))
    diseased_weight = 1 - severity_weight
    # batch x 1
    diseased_avg_reward = torch.mean(diseased_action_rewards, dim=-1, keepdim=True)
    valid_severity_avg_reward = torch.mean(severity_action_rewards, dim=-1, keepdim=True)
    action_rewards = diseased_weight * (diseased_action_rewards - diseased_avg_reward) + \
                     severity_weight * (severity_action_rewards - valid_severity_avg_reward)

    if args.greedy_action:
        # Local baseline
        loss = -1 * (action_logprobs * action_rewards) / (actions.size(-1) - 1)
        # batch
        loss = loss.sum(dim=1)
        # Average over batch
        # Divide by batches_step to mimic taking mean over larger batch
        loss = loss.mean() / args.batches_step  # For consistency: we generally set batches_step to 1 for greedy
        loss.backward()

        # For greedy: initialise next step by randomly picking one of the measurements for every slice
        idx = random.randint(0, mask.shape[1] - 1)
        mask = mask[:, idx, :, :, :].unsqueeze(1)
        masked_kspace = masked_kspace[:, idx, :, :, :].unsqueeze(1)
        image_input = image_input[:, idx, :, :, :].unsqueeze(1)
        final_outputs = [outputs[0][:, idx, :], outputs[1][:, idx, :]]

    elif step != args.acquisition_steps - 1:  # Non-greedy but don't have full return yet.
        loss = torch.zeros(1)  # For logging

    return loss, mask, masked_kspace, image_input, final_outputs



