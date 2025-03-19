import argparse
import yaml
import torch
import pytorch_lightning as pl
from pl_modules import FastMriDataModule, Binary_ClassificationModule, FineTuneBinaryModule
from data.masking import create_full_acquisition_mask
from data.transforms import DataTransform
from pathlib import Path
import time
import click
from lightning.pytorch.loggers import WandbLogger,CSVLogger

torch.set_float32_matmul_precision('medium')

def build_args(config_path):

    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)

    current_time = time.strftime("%m-%d_%H")
    args.default_root_dir = Path(args.log_path) / args.log_name / current_time
    args.run_name = current_time
    thresh_dict = {}

    single_label_name = args.label_names.split(",")
    for idx in range(len(single_label_name)):
        thresh_dict[single_label_name[idx]] = args.thresh[idx]
    args.thresh_dict = thresh_dict

    if not args.default_root_dir.exists():
        args.default_root_dir.mkdir(parents=True)

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=3,
            verbose=True,
            monitor=args.monitor,
            mode="max",
        )
    ]

    if args.ckpt_path == 'None':
        checkpoint_dir = Path(checkpoint_dir)
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
        if ckpt_list:
            args.ckpt_path = str(ckpt_list[-1])
            print(f"Checkpoint path set to: {args.ckpt_path}")
        else:
            print("No checkpoint files found in the specified directory.")
            args.ckpt_path = None

    return args

@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option("--seed", required=True, type=int, help="Random Seed.")
@click.option("--initial_accelerations", required=True, type=int, help="initial_accelerations.")
@click.option("--final_accelerations", required=True, type=int, help="final_accelerations.")


def main(config_path,seed,initial_accelerations,final_accelerations):
    args = build_args(config_path)
    args.seed = seed
    args.initial_accelerations = initial_accelerations
    args.final_accelerations = final_accelerations
    pl.seed_everything(args.seed)

    # * data
    # masking
    mask = create_full_acquisition_mask(args.initial_accelerations, args.center_fractions, args.final_accelerations,
                                        args.mask_type, args.seed)
    # data transform
    train_transform = DataTransform(mask_func=mask,  kspace_size=args.kspace_size, recon_size=args.recon_size)
    val_transform = DataTransform(mask_func=mask, kspace_size=args.kspace_size, recon_size=args.recon_size)
    test_transform = DataTransform(mask_func=mask, kspace_size=args.kspace_size, recon_size=args.recon_size)

    # pl data module
    data_module = FastMriDataModule(
        list_path=args.list_path,
        data_path=args.data_path,
        label_names=args.label_names,
        class_list=args.class_list,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # * model
    if args.num_label == 1:
        if args.num_classes == 2:
            if args.fine_tune:
                model = FineTuneBinaryModule(
                    args=args
                )
            else:
                model = Binary_ClassificationModule(
                    args=args
                )

    logger = WandbLogger(project=args.log_name, name=args.run_name, save_dir=args.default_root_dir)
    # In the main function, after creating the WandB logger:
    csv_logger = CSVLogger(save_dir=args.default_root_dir, name=f"{seed}_csv_logs")

    # And then use both loggers when initializing the trainer:
    trainer = pl.Trainer(
        logger=[logger, csv_logger],  # List of loggers
        callbacks=args.callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        num_sanity_val_steps=0,
    )

    # * run
    if args.mode == 'train':
        if args.resume_train:
            trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
        else:
            trainer.fit(model, data_module)
    elif args.mode == 'val':
        trainer.validate(model, data_module, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')


if __name__ == '__main__':
    main()
