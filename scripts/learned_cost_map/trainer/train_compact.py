import argparse
import yaml
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostFourierVelModel
from learned_cost_map.trainer.utils import get_balanced_dataloaders, preprocess_data, avg_dict, get_FFM_freqs

import wandb
import time

USE_WANDB = True
print("Start of trainer file!")

def traversability_cost_loss(model, input, labels, mean_cost=None):
    pred_cost = model(input)
    criterion = nn.MSELoss(reduction="sum")
    pred_cost = pred_cost.squeeze()
    # Get loss averaged accross batch
    loss = criterion(pred_cost, labels)/pred_cost.shape[0]

    random_cost = torch.rand(pred_cost.shape).cuda()
    random_loss = criterion(random_cost, labels)/random_cost.shape[0]

    return_dict = OrderedDict(loss=loss, random_loss=random_loss)

    if mean_cost is not None:
        mean_cost = mean_cost * torch.ones(pred_cost.shape).cuda()
        mean_loss = criterion(mean_cost, labels)/mean_cost.shape[0]
        return_dict = OrderedDict(loss=loss, random_loss=random_loss, mean_loss=mean_loss)
    return loss, return_dict


def run_train_epoch(model, model_name, train_loader, optimizer, scheduler, grad_clip=None, fourier_freqs=None):
    model.train()
    all_metrics = []
    curr_lr = scheduler.get_last_lr()[0]
    for i, data_dict in enumerate(train_loader):
        print(f"Training batch {i}/{len(train_loader)}")
        input, labels = preprocess_data(data_dict, fourier_freqs)
        loss, _metric = traversability_cost_loss(model, input, labels)
        _metric["lr"] = torch.Tensor([curr_lr])
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        data_dict = None ## This might help with script taking too much memory
    scheduler.step()
    return avg_dict(all_metrics)

def get_val_metrics(model, model_name, val_loader, fourier_freqs=None):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for i,data_dict in enumerate(val_loader):
            print(f"Validation batch {i}/{len(val_loader)}")
            x, y = preprocess_data(data_dict, fourier_freqs)
            loss, _metric = traversability_cost_loss(model, x, y)
            all_metrics.append(_metric)
            data_dict = None ## This might help with script taking too much memory
    return avg_dict(all_metrics)


def main(model_name, models_dir, log_dir, map_config, num_epochs = 20, batch_size = 256, embedding_size = 512, mlp_size = 512, num_freqs=16, seq_length = 1, grad_clip = None, lr = 1e-3, gamma=1, weight_decay=0.0, eval_interval = 5, save_interval = 5, data_root_dir=None, train_split=None, val_split=None, balanced_loader=False, train_lc_dir=None, train_hc_dir=None, val_lc_dir=None, val_hc_dir=None, num_workers=4, shuffle_train=False, shuffle_val=False, multiple_gpus=False, pretrained=False, augment_data=False, high_cost_prob=None, fourier_scale=10.0, fine_tune=False, saved_model=None, saved_freqs=None, wanda=False, just_eval=False):

    print("\n=====\nROBOT IS ATV\n=====\n")
    ## Obtain DataLoaders

    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    print("Getting data loaders")
    time_data = time.time()
    print("Using ATV balanced loader")
    train_loader, val_loader = get_balanced_dataloaders(batch_size, data_root_dir, train_lc_dir, train_hc_dir, val_lc_dir, val_hc_dir, map_config, augment_data=augment_data, high_cost_prob=high_cost_prob)

    print(f"Got data loaders. {time.time()-time_data}")

    ## Set up model
    fourier_freqs = None
    model = CostFourierVelModel(input_channels=8, ff_size=num_freqs, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1, pretrained=pretrained)

    fourier_freqs = get_FFM_freqs(1, scale=fourier_scale, num_features=num_freqs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)


    if USE_WANDB:
        wandb.login(key="b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6")
        config = {
            'model_name': model_name,
            'log_dir': log_dir,
            'map_metadata': map_metadata,
            'crop_params': crop_params,
            'batch_size': batch_size,
            'embedding_size': embedding_size, 
            'mlp_size': mlp_size,
            'seq_length': seq_length,
            'lr': lr,
            'gamma':gamma,
            'weight_decay':weight_decay,
            'grad_clip': grad_clip,
            'num_epochs': num_epochs,
            'eval_interval': eval_interval,
            "pretrained": pretrained,
            'balanced_loader': balanced_loader,
            'augment_data': augment_data,
            'high_cost_prob': high_cost_prob,
            'fourier_scale': fourier_scale,
        }
        print("Training configuration: ")
        print(config)
        print("Setting up wandb init")
        wandb.init(project="SARA", reinit=True, config=config, settings=wandb.Settings(start_method='fork'))
        # wandb.init(project="SARA", reinit=True, config=config, settings=wandb.Settings(start_method='thread'))
        # wandb.init(project="SARA", config=config)
        print("Done setting up wandb init")

    print("Starting epochs loop")
    for epoch in range(num_epochs):
        if not just_eval:
            print(f"Training, epoch {epoch}")
            train_time = time.time()
            train_metrics = run_train_epoch(model, model_name, train_loader, optimizer, scheduler, grad_clip, fourier_freqs)
            print(f"Training epoch: {time.time()-train_time} s")
        print(f"Validation, epoch {epoch}")
        val_time = time.time()
        val_metrics = get_val_metrics(model, model_name,    val_loader, fourier_freqs)
        print(f"Validation epoch: {time.time()-val_time} s")

        #TODO : add plotting code for metrics (required for multiple parts)
        if USE_WANDB:
            if not just_eval:
                train_metrics['epoch'] = epoch
                train_metrics = {"train/"+k:v for k,v in train_metrics.items()}
                wandb.log(data=train_metrics, step=epoch)
            val_metrics['epoch'] = epoch
            val_metrics = {"validation/"+k:v for k,v in val_metrics.items()}
            wandb.log(data=val_metrics, step=epoch)

        if (epoch+1)%eval_interval == 0:
            if not just_eval:
                print(epoch, train_metrics)
            print(epoch, val_metrics)

        if ((epoch+1)%save_interval == 0) and (not just_eval):
            train_models_dir = os.path.join(models_dir, log_dir)
            if not os.path.exists(train_models_dir):
                os.makedirs(train_models_dir)
            save_dir = os.path.join(train_models_dir, f"epoch_{epoch+1}.pt")
            if fourier_freqs is not None:
                freqs_dir = os.path.join(train_models_dir, f"fourier_freqs.pt")
                torch.save(fourier_freqs.cpu(), freqs_dir)
            torch.save(model.state_dict(), save_dir)




if __name__ == '__main__':
    print("Inside main function")


    model = "CostFourierVelModel"
    models_dir = "/ocean/projects/cis220039p/guamanca/projects/learned_cost_map/models"
    log_dir = "train_psc"
    map_config = "/ocean/projects/cis220039p/guamanca/projects/learned_cost_map/configs/map_params.yaml"
    num_epochs = 50
    batch_size = 1024
    embedding_size = 512
    mlp_size = 512
    num_freqs = 8
    learning_rate = 0.0003
    gamma = 0.99
    weight_decay = 0.0000001
    eval_interval = 1
    save_interval = 1
    data_dir = "/ocean/projects/cis220039p/shared/tartancost/tartancost_data_2022"
    balanced_loader = True
    train_lc_dir = "lowcost_merged"
    train_hc_dir = "highcost_merged"
    val_lc_dir = "lowcost_val_merged"
    val_hc_dir = "highcost_val_merged"
    num_workers = 1
    augment_data = True
    fourier_scale = 10.0


    # Run training loop
    main(model_name=model,
         models_dir=models_dir,
         log_dir=log_dir, 
         map_config=map_config,
         num_epochs = num_epochs, 
         batch_size = batch_size, 
         embedding_size = embedding_size,
         mlp_size = mlp_size,
         num_freqs = num_freqs, 
         lr = learning_rate,
         gamma = gamma,
         weight_decay = weight_decay, 
         eval_interval = eval_interval, 
         save_interval = save_interval, 
         data_root_dir = data_dir, 
         balanced_loader = balanced_loader,
         train_lc_dir = train_lc_dir,
         train_hc_dir = train_hc_dir,
         val_lc_dir = val_lc_dir,
         val_hc_dir = val_hc_dir,
         num_workers = num_workers, 
         augment_data = augment_data, 
         fourier_scale = fourier_scale,
         )