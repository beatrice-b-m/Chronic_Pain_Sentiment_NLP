import pandas as pd
import random
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from itertools import chain
# import glob
import demoji

@dataclass
class FrameParams:
    df: pd.DataFrame
    class_name: str
    class_val: float
    

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

def remove_emojis(text):
    return demoji.replace(text,'') #remove emoji

def load_data_to_loader_dict(df_path, label_col: str, test_size: float = 0.2, 
                             validation: bool = True, seed: int = 13, 
                             max_len: int = 128, batch_size: int = 256):
    # load dataframe
    df = pd.read_csv(df_path)
    
    # remove emojis
    df['text'] = df['text'].apply(remove_emojis)
    
    # split df by label and send to frame param objects
    neg_df = FrameParams(
        df=df[df[label_col] == -1], 
        class_name='negative', 
        class_val=0
    )
    neu_df = FrameParams(
        df=df[df[label_col] == 0], 
        class_name='neutral', 
        class_val=1
    )
    pos_df = FrameParams(
        df=df[df[label_col] == 1], 
        class_name='positive', 
        class_val=2
    )

    # split dataframes
    df_list = [neg_df, neu_df, pos_df]
    train_df, val_df, test_df = split_n_dataframes(
        df_list, 
        id_var='tweet_id', 
        test_prop=test_size, 
        seed=seed, 
        validation=validation
    )

    print('train size:', len(train_df))
    print('val size:', len(val_df))
    print('test size:', len(test_df))

    print(f'\ntrain distribution:\n{train_df.class_label.value_counts(dropna=False, normalize=True)}')
    print(f'\nval distribution:\n{val_df.class_label.value_counts(dropna=False, normalize=True)}')
    print(f'\ntest distribution:\n{test_df.class_label.value_counts(dropna=False, normalize=True)}')

    # load roberta base as a tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        'roberta-base', 
        truncation=True, 
        do_lower_case=True
    )

    # load dataframes into dataset objects
    train_ds = TweetDataset(train_df, tokenizer, max_len)
    val_ds = TweetDataset(val_df, tokenizer, max_len)
    test_ds = TweetDataset(test_df, tokenizer, max_len)

    # load datasets into loaders
    train_loader = get_dataloader(train_ds, batch_size)
    val_loader = get_dataloader(val_ds, batch_size)
    test_loader = get_dataloader(test_ds, batch_size)
    
    loader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    return loader_dict