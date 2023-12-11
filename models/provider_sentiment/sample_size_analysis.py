import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# !pip install demoji
import demoji
import pandas as pd
import random
from dataclasses import dataclass
import numpy as np
import torch
# import seaborn as sns
import transformers
import json
import glob
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, F1Score, Precision, Recall
from itertools import chain
import contextlib

# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.metrics import classification_report, confusion_matrix

def seed_script(seed: int):
    # set torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # set numpy seed
    np.random.seed(seed)
    print("seed set...")
    
SEED = 13
seed_script(SEED)

# get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")

@dataclass
class FrameParams:
    df: pd.DataFrame
    class_name: str
    class_val: float
    
def remove_emojis(text):
    return demoji.replace(text,'') #remove emoji

# function to get the set of unique patient ids in the dataframe
# then split based on the train/val/test proportion
def split_ids(id_col, test_prop, validation, seed):
    # get set of unique ids and convert to a list
    id_list = list(set(id_col))

    # shuffle id list
    random.Random(seed).shuffle(id_list)

    # get split lengths
    id_list_len = len(id_list)

    # get the length of indexes to add to the train/test sets
    train_prop = 1.0 - (2 * test_prop)
    train_len = int(train_prop * id_list_len)
    test_len = int(test_prop * id_list_len)

    # index set ids
    if validation:
        train_ids = id_list[:train_len]
        val_ids = id_list[train_len:train_len+test_len]

    else:
        train_ids = id_list[:train_len+test_len]
        val_ids = None

    test_ids = id_list[train_len+test_len:]

    print('total ids:', id_list_len)

    print('train ids: {}, prop: {:.3f}'.format(
        len(train_ids),
        len(train_ids) / id_list_len
    ))

    if validation:
        print('val ids: {}, prop: {:.3f}'.format(
            len(val_ids),
            len(val_ids) / id_list_len
        ))

    print('test ids: {}, prop: {:.3f}\n'.format(
        len(test_ids),
        len(test_ids) / id_list_len
    ))

    return train_ids, val_ids, test_ids

# function to index pos/neg dataframes by set patient ids and merge them
def index_dataframes(df_obj_list, ids, id_var):
    # zip pos/neg dataframes and ids
    components = zip([df_obj.df for df_obj in df_obj_list], ids)

    # index dataframes by ids for pos/neg
    df_list = [df[df[id_var].isin(ids)] for df, ids in components]

    # merge pos/neg dataframes
    out_df = pd.concat(df_list, axis=0)
    return out_df

# function to split a positive and negative dataframe into train/val/test
# then merge positive and negative for each
def split_n_dataframes(df_list, id_var: str = 'tweet_id',
                       test_prop: float = 0.2, seed: int = 13,
                       validation: bool = True, label_col: str = 'label'):
    # add label columns to dataframes
    for df_obj in df_list:
        df_obj.df.loc[:, 'class_label'] = df_obj.class_val

    # get empty list to put dataframe set IDs
    df_ids = []

    # get ids for each split dataframe
    for df_obj in df_list:
        train_ids, val_ids, test_ids = split_ids(
            df_obj.df[id_var],
            test_prop,
            validation,
            seed
        )
        df_ids.append([train_ids, val_ids, test_ids])

    # transpose list to get sublists of all train set IDs, val sets IDs, etc.
    trans_df_ids = [i for i in zip(*df_ids)]

    # prepare lists for indexing
    train_ids = trans_df_ids[0]
    val_ids = trans_df_ids[1]
    test_ids = trans_df_ids[2]

    # index split dataframes
    train_df = index_dataframes(df_list, train_ids, id_var)
    test_df = index_dataframes(df_list, test_ids, id_var)
    if validation:
        val_df = index_dataframes(df_list, val_ids, id_var)

    # shuffle dataframes
    train_df = train_df.sample(frac=1, random_state=seed).reset_index()
    test_df = test_df.sample(frac=1, random_state=seed).reset_index()
    if validation:
        val_df = val_df.sample(frac=1, random_state=seed).reset_index()
    else:
        val_df = None

    return train_df, val_df, test_df

def k_fold_generator(df, k):
    """
    function to split a dataframe into k folds and return an iterator
    """
    # shuffle the dataframe
    shuff_df = df.sample(frac=1).reset_index(drop=True)
    
    # split dataframe into a list of k folds of (near) equal size
    fold_list = np.array_split(shuff_df, k)
    
    # build a k length list of Trues
    base_mask = [True]*k
    
    # enumerate folds
    for i, fold in enumerate(fold_list):
        # copy base mask and set the ith index to False
        train_mask = base_mask.copy()
        train_mask[i] = False
        
        # set current chunk as the current validation df
        val_df = fold
        
        # apply the train mask to the chunk list and concatenate the
        # included folds
        train_df = pd.concat(
            [f for f, include in zip(fold_list, train_mask) if include], 
            axis=0
        )
        yield train_df, val_df
        
class TweetDataset(Dataset):
    """
    class is very closely based on the huggingface tutorial implementation
    """
    def __init__(self, dataframe, tokenizer, max_len, id_col: str = 'tweet_id',
                 text_col: str = 'text', target_col: str = 'class_label'):
        self.tokenizer = tokenizer
        # self.data = dataframe
        self.tweet_id_list = list(dataframe[id_col])
        self.text_list = list(dataframe[text_col])
        self.label_list = list(dataframe[target_col])
        self.max_len = max_len

    def __len__(self):
        # get length of dataset (required for dataloader)
        return len(self.text_list)

    def __getitem__(self, idx):
        # extract text
        text = str(self.text_list[idx])

        # extract label
        label = self.label_list[idx]

        # tokenize text
        encoded_text = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        # unpack encoded text
        ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        token_type_ids = encoded_text["token_type_ids"]

        # wrap outputs in dict
        out_dict = {
            'tweet_id_list': self.tweet_id_list,
            'id_tensor': torch.tensor(ids, dtype=torch.long),
            'mask_tensor': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_tensor': torch.tensor(token_type_ids, dtype=torch.long),
            'label_tensor': torch.tensor(label, dtype=torch.long)
        }

        return out_dict
    
def get_dataloader(dataset, batch_size, shuffle: bool = True,
                   pin_memory: bool = True, num_workers: int = 0,
                   prefetch_factor: int or None = None):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    return dataloader

class CustomRoberta(torch.nn.Module):
    """
    model subclass to define the RoBERTa architecture, also closely based on
    the huggingface tutorial implementation
    """
    def __init__(self, drop_percent, num_classes, pt_model_name: str = 'roberta-base'):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(pt_model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(drop_percent)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # get outputs from base model
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # extract hidden state from roberta base outputs
        hidden_state = base_outputs[0]
        x = hidden_state[:, 0]

        # define the linear layer preceding the classifier
        # and apply ReLU activation to its outputs
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)

        # define the dropout layer and classifier
        # and apply Softmax activation to its outputs
        x = self.dropout(x)
        x = self.classifier(x)
        # outputs = torch.nn.Softmax(dim=-1)(x)
        return x #outputs
    
def train_model(model, loader_dict, metric_collection, 
                criterion, optimizer, save_dir: str or None = None, 
                num_epochs: int = 25, monitor_metric: str = 'val_loss', 
                disable_bar: bool = False):
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
            with tqdm(phase_loader, unit="batch", total=phase_size, disable=disable_bar) as epoch_iter:
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
                        preds = model(
                            id_tensor,
                            mask_tensor,
                            token_type_tensor
                        )
                        # preds = outputs
                                                
                        # preds = torch.squeeze(outputs)
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
    
    # id_list, pred_list, label_list = evaluate_model(model, loader_dict['test'], metric_collection, criterion)
    # return id_list, pred_list, label_list
    test_metrics_dict = evaluate_model(model, loader_dict['test'], metric_collection, criterion, disable_bar)
    return test_metrics_dict

def evaluate_model(model, test_loader, metric_collection, criterion, disable_bar):
    running_loss = 0.0
    
    tweet_id_list = []
    pred_list = []
    label_list = []

    phase_size = len(test_loader)

    # iterate over data in current phase loader
    with tqdm(test_loader, unit="batch", total=phase_size, disable=disable_bar) as epoch_iter:
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
                preds = model(
                    id_tensor,
                    mask_tensor,
                    token_type_tensor
                )
                # preds = outputs # torch.squeeze(outputs)
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
        
    return phase_metrics_dict
    # return tweet_id_list, pred_list, label_list

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

def average_dicts(dict_list):
    # Initialize a dictionary to store the sum of values for each key
    sum_dict = {key: 0.0 for key in dict_list[0].keys()}

    # Sum up values for each key
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value

    # Calculate the average for each key
    avg_dict = {key: value / len(dict_list) for key, value in sum_dict.items()}

    return avg_dict

def train_eval_run(df_dict, device, tokenizer, param_dict: dict, max_len: int = 128, batch_size: int = 256, n_classes: int = 3):
    # load dataframes into dataset objects
    train_ds = TweetDataset(df_dict['train'], tokenizer, max_len)
    val_ds = TweetDataset(df_dict['val'], tokenizer, max_len)
    test_ds = TweetDataset(df_dict['test'], tokenizer, max_len)

    # load datasets into loaders
    train_loader = get_dataloader(train_ds, batch_size)
    val_loader = get_dataloader(val_ds, batch_size)
    test_loader = get_dataloader(test_ds, batch_size)
    
    # build metric collection and send it to gpu
    metric_collection = MetricCollection({
        'acc': Accuracy(task='multiclass', num_classes=n_classes),
        'auc': AUROC(task='multiclass', num_classes=n_classes),
        'prec': Precision(task='multiclass', num_classes=n_classes),
        'rec': Recall(task='multiclass', num_classes=n_classes),
        'micro_f1': F1Score(task='multiclass', num_classes=n_classes, average='micro'),
        'macro_f1': F1Score(task='multiclass', num_classes=n_classes, average='macro')
    })
    metric_collection.to(device)

    # build the model and send it to gpu
    model = CustomRoberta(param_dict['dropout_proportion'], n_classes)
    model.to(device)

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
    
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    test_metrics_dict = train_model(
        model, 
        loader_dict, 
        metric_collection, 
        criterion, 
        optimizer, 
        save_dir="./train_size_temp", 
        num_epochs=40, 
        monitor_metric='val_micro_f1',
        disable_bar=True
    )
    return test_metrics_dict

file_path = "/opt/localdata/Data/bea/nlp/bmi550/project/sentiment_data/combined_tweets.csv"
df = pd.read_csv(file_path)

# remove emojis from tweets
df['text'] = df['text'].apply(remove_emojis)

# build frameparam objects for each class
neg_df = FrameParams(
    df=df[df['provider_sentiment'] == -1], 
    class_name='negative', 
    class_val=0
)
neu_df = FrameParams(
    df=df[df['provider_sentiment'] == 0], 
    class_name='neutral', 
    class_val=1
)
pos_df = FrameParams(
    df=df[df['provider_sentiment'] == 1], 
    class_name='positive', 
    class_val=2
)

# split dataframes to train_val/test
df_list = [neg_df, neu_df, pos_df]
train_val_df, _, test_df = split_n_dataframes(
    df_list, 
    id_var='tweet_id', 
    test_prop=0.2, 
    seed=13, 
    validation=False
)

print('train_val size:', len(train_val_df))
# print('val size:', len(val_df))
print('test size:', len(test_df))

print(f'\ntrain/val distribution:\n{train_val_df.class_label.value_counts(dropna=False, normalize=True)}')
print(f'\ntest distribution:\n{test_df.class_label.value_counts(dropna=False, normalize=True)}')

# extract the best performing parameters based on the monitor metric
log_path = "/opt/localdata/Data/bea/nlp/bmi550/project/Chronic_Pain_Sentiment_NLP/models/provider_sentiment/logs/roberta_opt_2.json"
# open the log file
opt_df = pd.read_json(log_path, lines=True)

best_idx = opt_df["test_f1"].idxmax()
best_param_dict = opt_df.loc[best_idx, 'params']

# load roberta base as a tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

train_size_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
n_iter = 10

eval_metrics_list = []

# iterate over training set sizes
for curr_train_size in train_size_list:
    # train_size_metrics_list = []
    
    # repeat each for n iterations for bootstrapping
    for idx in range(n_iter):
        print(f"[train_size: {curr_train_size}, it: {idx}] {'-'*40}")
        
        fold_metrics_list = []
        
        # randomly sample from train_val_df
        train_val_sample_df = train_val_df.sample(curr_train_size)
        
        # get k-fold generator
        fold_gen = k_fold_generator(train_val_sample_df, 5)
        
        # iterate over k-fold generator
        for train_df, val_df in fold_gen:
            # build loader dict
            df_dict = {
                'train': train_df, 
                'val': val_df, 
                'test': test_df
            }
            
            # silence function prints
            # from https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # start train/eval run and retrieve metrics
                    test_metrics_dict = train_eval_run(
                        df_dict, 
                        device, 
                        tokenizer, 
                        best_param_dict
                    )
            
            # add test metrics to the current fold list
            fold_metrics_list.append(test_metrics_dict)
            
        # average all metrics from the k-fold validation
        avg_fold_metrics_dict = average_dicts(fold_metrics_list)
        avg_fold_metrics_dict['train_size'] = curr_train_size
        avg_fold_metrics_dict['iteration'] = idx
        
        eval_metrics_list.append(avg_fold_metrics_dict)
        
        # # add iteration metrics to the current train size list
        # train_size_metrics_list.append(avg_fold_metrics_dict)
        
    # # average all metrics from the n iterations
    # avg_size_metrics_dict = average_dicts(train_size_metrics_list)
    
    # add current train size to the dict and add it to the overall eval list
    # avg_size_metrics_dict['train_size'] = curr_train_size
    # eval_metrics_list.append(avg_size_metrics_dict)
    
eval_df = pd.DataFrame(eval_metrics_list)
eval_df.to_csv("./sample_size_analysis.csv", index=False)
print("df saved...")