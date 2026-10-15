"""
data re-construction for training and inference
"""
from pathlib import Path
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from copy import deepcopy

import pickle

from transformers import AutoModel, AutoTokenizer, BertTokenizer
import torch

from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, Dataset

import logging


class InputExample(Dataset):
    def __init__(self, encodings, labels, **kwargs):
        """
        sentences: List of dictionaries with keys 'anchor', 'positives', 'negatives'.
        tokenizer: Pretrained tokenizer.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class Input2Eval(Dataset):
    def __init__(self, encodings1, encodings2, ids, **kwargs):
        """
        sentences: List of dictionaries with keys 'anchor', 'positives', 'negatives'.
        tokenizer: Pretrained tokenizer.
        """
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "s1": {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()},
            "s2": {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()},
            "id": self.ids[idx]
        }
        return item

class DataProcessor(object):
    def __init__(self, args, **kwargs):
        self.args = args

    def load_data(self, subset="train"):
        """
        load from cache or create new
        """

        cache_pth = Path(self.args.out_dir).joinpath("cache-{}-{}".format(Path(self.args.model_name).stem, subset))
        if cache_pth.exists():
            logging.info("Loading cache from {}".format(cache_pth.as_posix()))
            return torch.load(cache_pth.as_posix())

        fpth = self.args.raw_dir

        df = pd.read_csv(fpth, sep="\t")
        df = df[df["set"]==subset]

        samples = df["sentence"].tolist()
        labels = df["label"].tolist()

        # tokenizer = AutoTokenizer.from_pretrained(self.args.model_name) # Using pre-trained model's tokenizer #! hellonlp的模型不能用autotokenizer，只能用berttokenizer
        tokenizer = BertTokenizer.from_pretrained(self.args.model_name) 
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True  
        tokenizer.model_max_length = self.args.max_length

        encodings = tokenizer(samples, truncation=True, padding=True)

        dataset = InputExample(
            encodings,
            labels
        )

        logging.info("Saving cache to  {}".format(cache_pth.as_posix()))
        torch.save(dataset, cache_pth.as_posix())
        
        return dataset

    def load_eval_data(self, subset="test"):
        """
        return sent pairs: id, pair
        return gts
        """
        df = pd.read_json(self.args.label_pth, lines=True)

        logging.info("set type: {}".format(subset))

        df = df[df["set"]==subset]

        cache_pth = Path(self.args.out_dir).joinpath("cache-{}-{}".format(Path(self.args.model_name).stem, subset))
        if cache_pth.exists():
            logging.info("Loading cache from {}".format(cache_pth.as_posix()))
            return torch.load(cache_pth.as_posix()), df
    

        tokenizer = BertTokenizer.from_pretrained(self.args.model_name) 
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True  
        tokenizer.model_max_length = self.args.max_length

        ids, p1s, p2s = [], [], []
        for _, row in df.iterrows():
            if pd.isna(row["sim"]):  # in
                assert not pd.isna(row["alt 2"]) and not pd.isna(row["otherID"])

                ids.append(row["ID"])
                p1s.append(row["sent"])
                p2s.append(row["alt 2"])

                ids.append(row["otherID"])
                p1s.append(row["alt 1"])
                p2s.append(row["alt 2"])

            else:  # c (s, a1)
                assert pd.isna(row["alt 2"])
                ids.append(row["ID"])
                p1s.append(row["sent"])
                p2s.append(row["alt 1"])
            
        assert len(ids) == len(p1s) == len(p2s)
    
        encodings1 = tokenizer(p1s, truncation=True, padding=True)
        encodings2 = tokenizer(p2s, truncation=True, padding=True)

        dataset = Input2Eval(encodings1, encodings2, ids)

        logging.info("Saving cache to  {}".format(cache_pth.as_posix()))
        torch.save(dataset, cache_pth.as_posix())

        return dataset, df

    @staticmethod
    def _triplet():
        pass

    @staticmethod
    def _npair():
        pass

    def _nt_xent(self, row):
        pass

def collate_fn(batch, tokenizer, max_length, **kwargs):
    """
    """
    pass


