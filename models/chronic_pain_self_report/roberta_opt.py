import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# seed torch operations
SEED = 13
torch.manual_seed(SEED)

# define hyperparameter ranges to search
hparam_range_dict = {
    'dropout_proportion': (0.001, 0.999), 'learning_rate': (0.00000001, 0.001)
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
hparam_optimizer.load_data("/opt/localdata/Data/bea/nlp/bmi550/project/chronic_pain_model_data/*.csv")

# start the optimization loop
n_random = 1
n_guided = 2
opt_idx = 'test'
hparam_optimizer.optimize(
    n_random=n_random, 
    n_guided=n_guided, 
    opt_idx=opt_idx
)