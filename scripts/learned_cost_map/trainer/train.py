import argparse
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel

from learned_cost_map.trainer.utils import *

import wandb
import time

USE_WANDB = True

def traversability_cost_loss(model, input, labels):
    pred_cost = model(input)
    criterion = nn.MSELoss(reduction="sum")
    pred_cost = pred_cost.squeeze()
    # Get loss averaged accross batch
    loss = criterion(pred_cost, labels)/pred_cost.shape[0]

    random_cost = torch.rand(pred_cost.shape).cuda()
    random_loss = criterion(random_cost, labels)/random_cost.shape[0]

    return loss, OrderedDict(loss=loss, random_loss=random_loss)

def run_train_epoch(model, train_loader, optimizer, scheduler, grad_clip=None, fourier_freqs=None):
    model.train()
    all_metrics = []
    curr_lr = scheduler.get_last_lr()[0]
    for i, data_dict in enumerate(train_loader):
        print(f"Training batch {i}/{len(train_loader)}")
        input, labels = preprocess_data(data_dict)

        loss, _metric = traversability_cost_loss(model, input, labels)
        _metric["lr"] = torch.Tensor([curr_lr])
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    scheduler.step()
    return avg_dict(all_metrics)

def get_val_metrics(model, val_loader, fourier_freqs=None):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for i,data_dict in enumerate(val_loader):
            print(f"Validation batch {i}/{len(val_loader)}")
            x, y = preprocess_data(data_dict)
            loss, _metric = traversability_cost_loss(model, x, y)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)


def main(model_name, log_dir, num_epochs = 20, batch_size = 256, seq_length = 1,
         grad_clip=None, lr = 1e-3, gamma=1, eval_interval = 5, save_interval = 5, saved_model=None, data_root_dir=None, train_split=None, val_split=None, num_workers=4, shuffle_train=False, shuffle_val=False, multiple_gpus=False):

    if (data_root_dir is None) or (train_split is None) or (val_split is None):
        raise NotImplementedError()

    ## Obtain DataLoaders
    print("Getting data loaders")
    time_data = time.time()
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val)
    print(f"Got data loaders. {time.time()-time_data}")

    ## Set up model
    if model_name=="CostModel":
        model = CostModel(input_channels=8, output_size=1)
        fourier_freqs = None
    elif model_name=="CostVelModel":
        model = CostVelModel(input_channels=8, embedding_size=512, output_size=1)
        fourier_freqs = None
    elif model_name=="CostFourierVelModel":
        model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
        fourier_freqs = get_FFM_freqs(1, scale=10.0, num_features=16)
    else:
        raise NotImplementedError()
    
    if multiple_gpus and torch.cuda.device_count() > 1:
        print("Using up to ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))

    if USE_WANDB:
        wandb.login(key="b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6")
        config = {
            'model_name': model_name,
            'log_dir': log_dir,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'lr': lr,
            'gamma':gamma,
            'grad_clip': grad_clip,
            'num_epochs': num_epochs,
            'eval_interval': eval_interval
        }
        print("Training configuration: ")
        print(config)
        wandb.init(project="SARA", reinit=True, config=config, settings=wandb.Settings(start_method='fork'))


    for epoch in range(num_epochs):
        print(f"Training, epoch {epoch}")
        train_time = time.time()
        train_metrics = run_train_epoch(model, train_loader, optimizer, scheduler, grad_clip, fourier_freqs)
        print(f"Training epoch: {time.time()-train_time} s")
        print(f"Validation, epoch {epoch}")
        val_time = time.time()
        val_metrics = get_val_metrics(model, val_loader, fourier_freqs)
        print(f"Validation epoch: {time.time()-val_time} s")

        #TODO : add plotting code for metrics (required for multiple parts)
        if USE_WANDB:
            train_metrics['epoch'] = epoch
            val_metrics['epoch'] = epoch
            train_metrics = {"train/"+k:v for k,v in train_metrics.items()}
            val_metrics = {"validation/"+k:v for k,v in val_metrics.items()}
            wandb.log(data=train_metrics, step=epoch)
            wandb.log(data=val_metrics, step=epoch)

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

        if (epoch+1)%save_interval == 0:
            models_dir = os.path.join("models", log_dir)
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            save_dir = os.path.join(models_dir, f"epoch_{epoch+1}.pt")
            if fourier_freqs is not None:
                freqs_dir = os.path.join(models_dir, f"fourier_freqs.pt")
                torch.save(fourier_freqs.cpu(), freqs_dir)
            torch.save(model.state_dict(), save_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['CostModel', 'CostVelModel', 'CostFourierVelModel'], default='CostModel')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--train_split', type=str, required=True, help='Path to the file that contains the training split text file.')
    parser.add_argument('--val_split', type=str, required=True, help='Path to the file that contains the validation split text file.')
    parser.add_argument('--log_dir', type=str, required=True, help='String for where the models will be saved.')
    parser.add_argument("-n", "--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=1, help="Length of sequence used for training. See TartanDriveDataset for more details.")
    parser.add_argument('--grad_clip', type=float, help='Max norm of gradients. Leave blank for no grad clipping')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=1.0, help="Value by which learning rate will be decreased at every epoch.")
    parser.add_argument("--eval_interval", type=int, default=1, help="How often to evaluate on validation set.")
    parser.add_argument("--save_interval", type=int, default=1, help="How often to save model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader.")
    parser.add_argument('--shuffle_train', action='store_true', help="Shuffle batches for training in the DataLoader.")
    parser.add_argument('--shuffle_val', action='store_true', help="Shuffle batches for validation in the DataLoader.")
    parser.add_argument('--multiple_gpus', action='store_true', help="Use multiple GPUs if they are available.")
    parser.set_defaults(shuffle_train=False, shuffle_val=False, multiple_gpus=False)
    args = parser.parse_args()

    print(f"grad_clip is {args.grad_clip}")
    print(f"learning rate is {args.learning_rate}")

    # Run training loop
    main(model_name=args.model,
         log_dir=args.log_dir, 
         num_epochs = args.num_epochs, 
         batch_size = args.batch_size, 
         seq_length = args.seq_length, 
         grad_clip=args.grad_clip, 
         lr = args.learning_rate,
         gamma=args.gamma, 
         eval_interval = args.eval_interval, 
         save_interval=args.save_interval, 
         data_root_dir=args.data_dir, 
         train_split=args.train_split, 
         val_split=args.val_split,
         num_workers=args.num_workers, 
         shuffle_train=args.shuffle_train, 
         shuffle_val=args.shuffle_val,
         multiple_gpus=args.multiple_gpus
         )