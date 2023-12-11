import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from training import train_model
from model import CustomRoberta
from data import load_data_to_loader_dict
import pandas as pd
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall

TEST_SIZE = 0.2
VALIDATION = True
SEED = 13
MAX_LEN = 256
BATCH_SIZE = 128

# seed torch operations
torch.manual_seed(SEED)

# load/split data and send it to a dict of data loaders
base_file_path = "/opt/localdata/Data/bea/nlp/bmi550/project/chronic_pain_model_data/*.csv"
loader_dict = load_data_to_loader_dict(
    base_file_path, 
    test_size=TEST_SIZE, 
    validation=VALIDATION, 
    seed=SEED, 
    max_len=MAX_LEN, 
    batch_size=BATCH_SIZE
)

# open the log file
log_path = "/opt/localdata/Data/bea/nlp/bmi550/project/Chronic_Pain_Sentiment_NLP/models/chronic_pain_self_report/logs/roberta_opt_2.json"
opt_df = pd.read_json(log_path, lines=True)

# # extract the best performing parameters based on the monitor metric
# best_idx = opt_df["test_f1"].idxmax()
# best_param_dict = opt_df.loc[best_idx, 'params']

best_param_dict = {"dropout_proportion": 0.3, "learning_rate": 0.000005}

# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")

# build the model and send it to the gpu
model = CustomRoberta(best_param_dict['dropout_proportion'], 1)
model.to(device)

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
metric_collection.to(device)

# define loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=best_param_dict['learning_rate']
)

eval_metric_dict = train_model(
    model,
    device,
    loader_dict, 
    metric_collection, 
    criterion, 
    optimizer, 
    save_dir="./final_model/", 
    num_epochs=30, 
    monitor_metric="val_f1"
)