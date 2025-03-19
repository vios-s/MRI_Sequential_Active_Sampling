import torch
import sys
sys.path.append('../')
from models import MultiHeadResNet50
import pytorch_lightning as pl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_infer_model(args, task=None, optim=False):
    pl.seed_everything(args.seed)
    if task is None:
        ckpt_dir = args.infer_model_checkpoint
        checkpoint = torch.load(args.infer_model_checkpoint)
    elif task == 'diseased':
        ckpt_dir = args.diseased_infer_model_checkpoint
        checkpoint = torch.load(args.diseased_infer_model_checkpoint)
    elif task == 'severity':
        ckpt_dir = args.severity_infer_model_checkpoint
        checkpoint = torch.load(args.severity_infer_model_checkpoint)
    else:
        print(f'No target task named {task}!')
    infer_model = MultiHeadResNet50(args)

    new_state_dict = {}

    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("model.", "")  # Remove the "model." prefix
        new_state_dict[new_key] = v
        if new_key == 'criterion.weight':
            del new_state_dict['criterion.weight']

    if not optim:
        # No gradients for this model
        for param in infer_model.parameters():
            param.requires_grad = False

    infer_model.load_state_dict(new_state_dict, strict=False)

    # Set the model to evaluation mode
    infer_model.eval()
    infer_model.backbone.eval()

    del checkpoint

    print('\n')
    print(f'Successfully load checkpoint from {ckpt_dir}!!')

    return infer_model