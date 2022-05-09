import argparse
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel

from learned_cost_map.trainer.utils import *

import wandb

USE_WANDB = True

def traversability_cost_loss(model, x, y):
    pred_cost = model(x)
    criterion = nn.MSELoss(reduction="sum")
    pred_cost = pred_cost.squeeze()
    # Get loss averaged accross batch
    loss = criterion(pred_cost, y)/pred_cost.shape[0]

    random_cost = torch.rand(pred_cost.shape).cuda()
    random_loss = criterion(random_cost, y)/random_cost.shape[0]

    return loss, OrderedDict(loss=loss, random_loss=random_loss)

def run_train_epoch(model, train_loader, optimizer, grad_clip = None):
    model.train()
    all_metrics = []
    for i, data_dict in enumerate(train_loader):
        print(f"Training batch {i}/{len(train_loader)}")
        x, y = preprocess_data(data_dict)

        loss, _metric = traversability_cost_loss(model, x, y)
        print(f"Loss: {loss}")
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)

def get_val_metrics(model, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for i,data_dict in enumerate(val_loader):
            print(f"Validation batch {i}/{len(val_loader)}")
            x, y = preprocess_data(data_dict)
            loss, _metric = traversability_cost_loss(model, x, y)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)


def main(log_dir, num_epochs = 20, batch_size = 256, seq_length = 10,
         grad_clip=None, lr = 1e-3, eval_interval = 5, save_interval = 5, saved_model=None, data_root_dir=None, train_split=None, val_split=None):
    if (data_root_dir is None) or (train_split is None) or (val_split is None):
        raise NotImplementedError()

    # os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split)

    model = CostModel(input_channels=8, output_size=1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))

    if USE_WANDB:
        wandb.login(key="b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6")
        config = {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'lr': lr,
            'grad_clip': grad_clip,
            'num_epochs': num_epochs,
            'eval_interval': eval_interval
        }
        print("Training configuration: ")
        print(config)
        wandb.init(project="SARA", reinit=True, config=config)


    for epoch in range(num_epochs):
        print(f"Training, epoch {epoch}")
        train_metrics = run_train_epoch(model, train_loader, optimizer, grad_clip)
        print(f"Validation, epoch {epoch}")
        val_metrics = get_val_metrics(model, val_loader)

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
            torch.save(model.state_dict(), save_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--train_split', type=str, required=True, help='Path to the file that contains the training split text file.')
    parser.add_argument('--val_split', type=str, required=True, help='Path to the file that contains the validation split text file.')
    args = parser.parse_args()

    # Run training loop
    main('sara_cluster', num_epochs = 50, batch_size = 16, seq_length = 10,
         grad_clip=None, lr = 1e-4, eval_interval = 1, save_interval=1, data_root_dir=args.data_dir, train_split=args.train_split, val_split=args.val_split)