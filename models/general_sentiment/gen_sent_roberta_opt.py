import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
from tuning import HyperParamOptimizer

monitor_metric = "f1"
epochs_per_run = 40
SEED = 13
n_random = 30
n_guided = 70
opt_idx = '3'

# define hyperparameter ranges to search
hparam_range_dict = {
    'dropout_proportion': (0.1, 0.9), 'learning_rate': (0.0000001, 0.0001)
}

# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")

# define optimizer
hparam_optimizer = HyperParamOptimizer(
    device=device, 
    hparam_range_dict=hparam_range_dict, 
    monitor_metric=monitor_metric, 
    epochs_per_run=epochs_per_run
)

# load data into optimizer
df_path = "/opt/localdata/Data/bea/nlp/bmi550/project/sentiment_data/combined_tweets.csv"
hparam_optimizer.load_data(df_path, 'general_sentiment')

# start the optimization loop
hparam_optimizer.optimize(
    n_random=n_random, 
    n_guided=n_guided, 
    opt_idx=opt_idx
)