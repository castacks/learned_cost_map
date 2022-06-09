import argparse
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel, CostModelEfficientNet, CostFourierVelModelEfficientNet

from learned_cost_map.trainer.utils import get_dataloaders, get_balanced_dataloaders, preprocess_data, avg_dict, get_FFM_freqs

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
        input, labels = preprocess_data(data_dict, fourier_freqs)

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
            x, y = preprocess_data(data_dict, fourier_freqs)
            loss, _metric = traversability_cost_loss(model, x, y)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)


def main(model_name, log_dir, num_epochs = 20, batch_size = 256, seq_length = 1,
         grad_clip=None, lr = 1e-3, gamma=1, weight_decay=0.0, eval_interval = 5, save_interval = 5, data_root_dir=None, train_split=None, val_split=None, balanced_loader=False, train_lc_dir=None, train_hc_dir=None, val_lc_dir=None, val_hc_dir=None, num_workers=4, shuffle_train=False, shuffle_val=False, multiple_gpus=False, pretrained=False, augment_data=False, high_cost_prob=None, fourier_scale=10.0, fine_tune=False, saved_model=None, saved_freqs=None):

    if (data_root_dir is None):
        raise NotImplementedError()

    ## Obtain DataLoaders
    print("Getting data loaders")
    time_data = time.time()
    if balanced_loader:
        assert ((train_lc_dir is not None) and (train_hc_dir is not None) and (val_lc_dir is not None) and (val_hc_dir is not None)), "balanced_loader needs train_lc_dir, train_hc_dir, val_lc_dir, val_hc_dir to NOT be None."

        train_loader, val_loader = get_balanced_dataloaders(batch_size, data_root_dir, train_lc_dir, train_hc_dir, val_lc_dir, val_hc_dir, augment_data=augment_data, high_cost_prob=high_cost_prob)
    else:
        assert ((train_split is not None) and (val_split is not None)), "Standard dataloader needs train_split, val_split to NOT be None."

        train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, augment_data=augment_data)
    print(f"Got data loaders. {time.time()-time_data}")

    ## Set up model
    fourier_freqs = None
    if model_name=="CostModel":
        model = CostModel(input_channels=8, output_size=1, pretrained=pretrained)
    elif model_name=="CostVelModel":
        model = CostVelModel(input_channels=8, embedding_size=512, output_size=1, pretrained=pretrained)
    elif model_name=="CostFourierVelModel":
        model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, output_size=1, pretrained=pretrained)
        if fine_tune:
            assert (saved_freqs is not None), "saved_freqs needs to be passed as input"
            fourier_freqs = torch.load(saved_freqs)
        else:
            fourier_freqs = get_FFM_freqs(1, scale=fourier_scale, num_features=16)
    elif model_name=="CostModelEfficientNet":
        model = CostModelEfficientNet(input_channels=8, output_size=1, pretrained=pretrained)
    elif model_name=="CostFourierVelModelEfficientNet":
        model = CostFourierVelModelEfficientNet(input_channels=8, ff_size=16, embedding_size=512, output_size=1, pretrained=pretrained)
        if fine_tune:
            assert (saved_freqs is not None), "saved_freqs needs to be passed as input"
            fourier_freqs = torch.load(saved_freqs)
        else:
            fourier_freqs = get_FFM_freqs(1, scale=fourier_scale, num_features=16)
    else:
        raise NotImplementedError()
    
    if fine_tune:
        assert (saved_model is not None), "saved_model needs to be passed as input"
        print(f"Loading the following model: {saved_model}")
        model.load_state_dict(torch.load(saved_model))
        print("Pre-trained model successfully loaded!")

    if multiple_gpus and torch.cuda.device_count() > 1:
        print("Using up to ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)


    if USE_WANDB:
        wandb.login(key="b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6")
        config = {
            'model_name': model_name,
            'log_dir': log_dir,
            'batch_size': batch_size,
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
            'fourier_scale': fourier_scale
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
    parser.add_argument('--model', choices=['CostModel', 'CostVelModel', 'CostFourierVelModel', 'CostModelEfficientNet', 'CostFourierVelModelEfficientNet'], default='CostModel')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--train_split', type=str, help='Path to the file that contains the training split text file.')
    parser.add_argument('--val_split', type=str, help='Path to the file that contains the validation split text file.')
    parser.add_argument('--log_dir', type=str, required=True, help='String for where the models will be saved.')
    parser.add_argument('--balanced_loader', action='store_true', help="Use the balanced dataloader implemented in TartanDriveBalancedDataset.")
    parser.add_argument('--train_lc_dir', type=str, help='Name of directory where the low cost training set is located. Relative to data_dir. Only required if balanced_loader flag is present.')
    parser.add_argument('--train_hc_dir', type=str, help='Name of directory where the high cost training set is located. Relative to data_dir. Only required if balanced_loader flag is present.')
    parser.add_argument('--val_lc_dir', type=str, help='Name of directory where the low cost validation set is located. Relative to data_dir. Only required if balanced_loader flag is present.')
    parser.add_argument('--val_hc_dir', type=str, help='Name of directory where the high cost validation set is located. Relative to data_dir. Only required if balanced_loader flag is present.')
    parser.add_argument("-n", "--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=1, help="Length of sequence used for training. See TartanDriveDataset for more details.")
    parser.add_argument('--grad_clip', type=float, help='Max norm of gradients. Leave blank for no grad clipping')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=1.0, help="Value by which learning rate will be decreased at every epoch.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="L2 penalty (default is 0.0).")
    parser.add_argument("--eval_interval", type=int, default=1, help="How often to evaluate on validation set.")
    parser.add_argument("--save_interval", type=int, default=1, help="How often to save model.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the DataLoader.")
    parser.add_argument('--shuffle_train', action='store_true', help="Shuffle batches for training in the DataLoader.")
    parser.add_argument('--shuffle_val', action='store_true', help="Shuffle batches for validation in the DataLoader.")
    parser.add_argument('--multiple_gpus', action='store_true', help="Use multiple GPUs if they are available.")
    parser.add_argument('--pretrained', action='store_true', help="Use pretrained ResNet.")
    parser.add_argument('--augment_data', action='store_true', help="Augment data.")
    parser.add_argument('--high_cost_prob', type=float, help="Probability of high cost frames in data. If not set, defaults to None, and balanced data split.")
    parser.add_argument('--fourier_scale', type=float, default=10.0, help="Scale for Fourier frequencies, only needed for CostFourierVel models. If not set, defaults to 10.0.")
    parser.add_argument('--fine_tune', action='store_true', help="Augment data.")
    parser.add_argument('--saved_model', type=str, help='String for where the saved model that will be used for fine tuning is located.')
    parser.add_argument('--saved_freqs', type=str, help='String for where the saved Fourier frequencies that will be used for fine tuning are located.')

    parser.set_defaults(balanced_loader=False, shuffle_train=False, shuffle_val=False, multiple_gpus=False, pretrained=False, augment_data=False, fine_tune=False)
    args = parser.parse_args()

    print(f"grad_clip is {args.grad_clip}")
    print(f"learning rate is {args.learning_rate}")
    print(f"pretrained is {args.pretrained}")
    print(f"weight decay is {args.weight_decay}")
    print(f"high_cost_prob is {args.high_cost_prob}")
    print(f"fine_tune is {args.fine_tune}")
    print(f"saved_model is {args.saved_model}")

    # Run training loop
    main(model_name=args.model,
         log_dir=args.log_dir, 
         num_epochs = args.num_epochs, 
         batch_size = args.batch_size, 
         seq_length = args.seq_length, 
         grad_clip=args.grad_clip, 
         lr = args.learning_rate,
         gamma=args.gamma,
         weight_decay=args.weight_decay, 
         eval_interval = args.eval_interval, 
         save_interval=args.save_interval, 
         data_root_dir=args.data_dir, 
         train_split=args.train_split, 
         val_split=args.val_split,
         balanced_loader=args.balanced_loader,
         train_lc_dir=args.train_lc_dir,
         train_hc_dir=args.train_hc_dir,
         val_lc_dir=args.val_lc_dir,
         val_hc_dir=args.val_hc_dir,
         num_workers=args.num_workers, 
         shuffle_train=args.shuffle_train, 
         shuffle_val=args.shuffle_val,
         multiple_gpus=args.multiple_gpus,
         pretrained=args.pretrained,
         augment_data=args.augment_data, 
         high_cost_prob=args.high_cost_prob,
         fourier_scale=args.fourier_scale,
         fine_tune=args.fine_tune,
         saved_model=args.saved_model,
         saved_freqs=args.saved_freqs
         )