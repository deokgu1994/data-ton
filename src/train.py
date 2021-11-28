import os
import sys
import glob
import argparse
import collections
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import pdb
def using_pdb():
    pdb.set_trace()

#
import adamp


# make by users
import data_loader.data_loaders as module_data
import data_loader.data_sets as module_data_set
import loss.loss as module_loss
import optimizer.metric as module_metric
import optimizer.optimizer as module_optimizer
import model.model as module_net
import transform.transform as module_transform
import trainer as Trainer
from parse_config import ConfigParser
from utils import prepare_device



def main(config):
    # fix random seeds for reproducibility
    SEED = config["seed"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    if config["save"]:
        logger = config.get_logger("train")

    device, device_ids = prepare_device(config["n_gpu"])
    print(f"device {device}, device_ids {device_ids}")

    # setup data_loader instances
    data_set = config.init_obj("data_set", module_data_set)
    if config["debug"]["set_debug"]:
        data_set.ratio = config["debug"]["ratio"]

    # load model
    model = config.init_obj("Net", module_net)
    # if config["save"] : logger.info(model)

    # set trasform
    transform = config.init_obj("transform", module_transform)
    
    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    data_set.set_transforms(transform.transforms)
    
    # get function handles of loss and metrics
    criterion = config.init_obj("loss", module_loss)

    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"]["type"] in dir(torch.optim):
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    elif config["optimizer"]["type"] in dir(adamp): # AdamP, SGDP
        optimizer = config.init_obj("optimizer", adamp, trainable_params)
    else:
        optimizer = config.init_obj("optimizer", module_optimizer, params = trainable_params)

    # # lr_scheduler
    # if config["lr_scheduler"]["type"] in dir(torch.optim.lr_scheduler):
    #     lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
    # else:
    #     assert print(f"lr_scheduler not load")
    lr_scheduler = None
    
    if config["type"] == "basic": 
        data_loader = DataLoader(data_set,
                            batch_size= config["batch_size"],
                            shuffle= config["shuffle"],
                            num_workers= config["num_workers"],
                            drop_last=True 
                        )

        if config["valid_set"]["is_pass"] == False:
            val_data_set = config.init_obj("valid_set", module_data_set)
            if config["debug"]["set_debug"]:
                val_data_set.ratio = config["debug"]["ratio"]
            val_data_set.set_transforms(transform.transforms)
            valid_data_loader = DataLoader(val_data_set,
                    batch_size= config["batch_size"],
                    shuffle= config["shuffle"],
                    num_workers= config["num_workers"],
                    drop_last=True 
                )
        else:
            valid_data_loader = None
        print(len(data_loader), len(valid_data_loader))
        trainer = Trainer.Trainer(
                        model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    elif config["type"] == "Classfication":
        trainer = Trainer.Trainer_cls(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            data_set=data_set,
            lr_scheduler=lr_scheduler,
        )
    elif config["type"] == "Segmentation":
        trainer = Trainer.Trainer_seg(
            model,
            criterion,
            optimizer,
            config=config,
            device=device,
            data_set=data_set,
            lr_scheduler=lr_scheduler,
        )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
