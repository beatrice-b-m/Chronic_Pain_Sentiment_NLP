import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import json
import logging
import random
from data import load_data_to_loader_dict
from training import train_model, evaluate_model
from model import CustomRoberta
# from model.base import *

class HyperParamOptimizer:
    def __init__(self, device, hparam_range_dict: dict, 
                 monitor_metric: str = 'loss', 
                 epochs_per_run: int = 30):
        # load model function and hyperparameter ranges
        self.hparam_range_dict = hparam_range_dict
        self.val_monitor_metric = f"val_{monitor_metric}"
        self.test_monitor_metric = f"test_{monitor_metric}"
        self.epochs_per_run = epochs_per_run
        
        # initialize parameters
        self.loader_dict = None
        self.target_feature = None
        self.seed = None
        
    def load_data(self, base_file_path):
        self.loader_dict = load_data_to_loader_dict(base_file_path=base_file_path)
        print("Data loaded...")
                
    def optimize(self, n_random, n_guided, opt_idx: str = '0'):
        # convert categorical hparams to continuous ranges
        hparam_dict = self.process_hparams()
        
        log_path = f"./logs/roberta_opt_{opt_idx}.json"
        
        # start the logger
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
        
        print(f"\nOptimizing {self.model_name} {'-'*60}")
        print(f"Logging results to '{log_path}'")

        # define optimizer
        optimizer = BayesianOptimization(
            f=self.objective_wrapper,
            pbounds=hparam_dict,
            random_state=self.seed)

        # maximize f1 with hyperparams
        optimizer.maximize(init_points=n_random, n_iter=n_guided)

        # open the log file
        opt_df = pd.read_json(log_path, lines=True)
        
        # extract the best performing parameters based on the micro-avg f1
        best_idx = opt_df[self.test_monitor_metric].idxmax()
        best_param_dict = opt_df.loc[best_idx, 'params']
        
        # train a model and evaluate it on the test set with the best performing parameters
        print('Run using best params: ', best_param_dict)
        # eval_obj = self.train_model(self.train_val_df, self.test_df, best_param_dict, verbose=False)
        
        metrics_dict = self.objective_function(**best_param_dict)
        
        # micro and macro average f1 are the same since we only have 1 test set/train set
        # log_data(best_param_dict, eval_obj.acc, eval_obj.f1, eval_obj.f1, final=True)
        
    def objective_wrapper(self, **kwargs):
        """
        wrapper to pass the keyword arguments from the bayes opt package
        to the objective function as a dict since we're passing the params 
        to the model_function (not the objective function) as a dict.
        """
        # hparam_dict = self.index_conv_categorical_params(kwargs)
        test_metric_dict = self.objective_function(**kwargs)
        eval_metric = test_metric_dict[self.test_monitor_metric]
        
        # log hyperparameter combination and eval metric (and if it's the final run)
        log_data(kwargs, eval_metric, final: bool = False)
        return eval_metric
        
    def objective_function(self, dropout_proportion, learning_rate):
        """
        params to optimize:
        dropout_proportion, learning_rate
        """
        
        # build the model and send it to the gpu
        model = CustomRoberta(dropout_proportion, 1)
        model.to(self.device)
        
        # define metric collection
        TASK_TYPE = 'binary'
        NUM_CLASSES = 2
        metric_collection = MetricCollection({
            'acc': Accuracy(task=TASK_TYPE, num_classes=NUM_CLASSES),
            'auc': AUROC(task=TASK_TYPE, num_classes=NUM_CLASSES),
            'prec': Precision(task=TASK_TYPE, num_classes=NUM_CLASSES),
            'rec': Recall(task=TASK_TYPE, num_classes=NUM_CLASSES),
            'f1': F1Score(task=TASK_TYPE, num_classes=NUM_CLASSES)
        })
        metric_collection.to(self.device)
        
        # define loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
        eval_metric_dict = train_model(
            model, 
            self.loader_dict, 
            metric_collection, 
            criterion, 
            optimizer, 
            save_dir="./tuning_temp/", 
            num_epochs=self.epochs_per_run, 
            monitor_metric=self.val_monitor_metric
        )
        
        return eval_metric_dict

def log_data(param_dict, metric_dict, final: bool = False):
    # build the output dict
    log_dict = {"final": final}
    log_dict.update(param_dict)
    log_dict.update(metric_dict)
    
    # convert the dict to a json string
    json_str = json.dumps(log_dict)
    
    # log the json string
    logging.info(json_str)
    print("Results logged...")
    
@dataclass
class Evaluation:
    acc: float
    prec: float
    rec: float
    f1: float