import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path


def train_model(model, device, loader_dict, metric_collection, 
                criterion, optimizer, save_dir: str or None = None, 
                num_epochs: int = 25, monitor_metric: str = 'val_loss'):
    if save_dir is not None:
        # if save dir doesn't exist, make it
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model_save_path = os.path.join(save_dir, 'best_model_params.pth')
    
    # save base weights
    torch.save(model.state_dict(), model_save_path)

    # initialize the best metric based on what the monitor metric is
    # (and if it should be maximized or minimized)
    if monitor_metric.split('_')[-1] == 'loss':
        best_metric = np.inf
    else:
        best_metric = -np.inf

    # iterate over epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch} {'-' * 40}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_size = 0

            # select current data loader
            phase_loader = loader_dict[phase]
            phase_size = len(phase_loader)

            # iterate over data in current phase loader
            with tqdm(phase_loader, unit="batch", total=phase_size) as epoch_iter:
                for batch, data in enumerate(epoch_iter):
                    # unpack data dict
                    id_tensor = data['id_tensor'].to(device)
                    mask_tensor = data['mask_tensor'].to(device)
                    token_type_tensor = data['token_type_tensor'].to(device)
                    label_tensor = data['label_tensor'].to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(
                            id_tensor,
                            mask_tensor,
                            token_type_tensor
                        )
                        preds = torch.squeeze(outputs)
                        loss = criterion(preds, label_tensor)

                        # update running loss
                        running_loss += loss.item() #* label_tensor.size(0)
                        # running_size += label_tensor.size(0)

                        # update metric collection
                        metric_collection.update(preds, label_tensor)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # update metrics after each 10% chunk
                    # or if in val update on last batch
                    if ((phase == 'train') & (batch % (max(phase_size // 10, 1)) == 0)) |\
                    ((phase == 'val') & (batch == (phase_size - 1))):
                        phase_metrics = metric_collection.compute()

                        phase_metrics_dict = format_metrics_dict(
                            loss, #/ running_size, 
                            phase_metrics, 
                            phase
                        )
                        epoch_iter.set_postfix(phase_metrics_dict)
                        
            # reset metric collection
            metric_collection.reset()
            
            # save the model weights if the current val monitor metric is the best so far
            if (save_dir is not None) & is_metric_better(monitor_metric, phase_metrics_dict, best_metric):
                best_metric = phase_metrics_dict[monitor_metric]
                
                print(f"saving model with best {monitor_metric} '{best_metric:.4f}'...")
                torch.save(model.state_dict(), model_save_path)

    # load best model weights and evaluate on test set
    model.load_state_dict(torch.load(model_save_path))
    test_metrics_dict = evaluate_model(model, device, loader_dict['test'], metric_collection, criterion)
    return test_metrics_dict

def evaluate_model(model, device, test_loader, metric_collection, criterion):
    running_loss = 0.0
    
    tweet_id_list = []
    pred_list = []
    label_list = []

    phase_size = len(test_loader)

    # iterate over data in current phase loader
    with tqdm(test_loader, unit="batch", total=phase_size) as epoch_iter:
        for batch, data in enumerate(epoch_iter):
            # unpack data dict
            batch_id_list = data['tweet_id_list']
            id_tensor = data['id_tensor'].to(device)
            mask_tensor = data['mask_tensor'].to(device)
            token_type_tensor = data['token_type_tensor'].to(device)
            label_tensor = data['label_tensor'].to(device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(
                    id_tensor,
                    mask_tensor,
                    token_type_tensor
                )
                preds = torch.squeeze(outputs)
                loss = criterion(preds, label_tensor)

                # update running loss
                running_loss += loss.item()

                # update metric collection
                metric_collection.update(preds, label_tensor)
                
                tweet_id_list += batch_id_list
                pred_list.append(preds.detach().cpu()) #.numpy())
                label_list.append(label_tensor.detach().cpu().numpy())

    phase_metrics = metric_collection.compute()

    phase_metrics_dict = format_metrics_dict(
        loss,
        phase_metrics, 
        'test'
    )

    # print metrics
    for k, v in phase_metrics_dict.items():
        print(f"{k}: {v:.4f}")
        
    return phase_metrics_dict # tweet_id_list, pred_list, label_list

def is_metric_better(monitor_metric, metrics_dict, best_eval):
    """
    function to determine if the monitor metric should be maximized or minimized
    """
    curr_eval = metrics_dict.get(monitor_metric)
    if curr_eval is None:
        return False
    
    if monitor_metric.split('_')[-1] == 'loss':
        return curr_eval < best_eval
    else:
        return curr_eval > best_eval
    
def format_metrics_dict(loss, metrics_dict, set_name: str):
    out_metrics_dict = {}
    out_metrics_dict[f'{set_name}_loss'] = loss.item()

    for k, v in metrics_dict.items():
        out_metrics_dict[f'{set_name}_{k}'] = v.item()

    return out_metrics_dict