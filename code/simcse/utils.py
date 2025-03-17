# -*- coding: utf-8 -*-
"""
"""
from pathlib import Path
import pandas as pd
import numpy as np

from random import randint
from tqdm import tqdm
from itertools import product
from scipy.stats import pearsonr, spearmanr

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # return self.text_2_id(self.data[index])
        return self.data[index]
     

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, encodings1, encodings2, ids, **kwargs):
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
    

class DataProcessor():
    def __init__(self, args, **kwargs):
        self.args = args


    @staticmethod
    def prepare_train_data(df, tokenizer):
        """
        create triplet (A, P, N)
        """

        sents = df.sent.drop_duplicates()
        # create (a, p, n) 
        samples = []
        pbar = tqdm(total=len(sents))
        for sent in sents:
            this = df[df["sent"]==sent]
            for idm in this["idiom"].drop_duplicates():
                if not idm:  # std sts
                    this2 = this[this["idiom"].isna()]
                    a1s = this2[this2['sim']==1]["alt 1"].drop_duplicates().tolist()
                    a2s = this2[this2["sim"]==0]["alt 1"].drop_duplicates().tolist()
                else:  # ists
                    this2 = this[this["idiom"]==idm]
                    a1s, a2s = this2["alt 1"].drop_duplicates().tolist(), this2["alt 2"].drop_duplicates().tolist()

                assert None not in a1s
                if None in a2s:
                    a2s.remove(None)

                # 没有neg sample，随机找neg
                a2s = [df.drop(this.index).sample(1)["sent"].item()] if not a2s else a2s
    
                # 没有pos sample，那就复制anchor
                a1s = [sent] if not a1s else a1s

                for (a1, a2) in product(a1s, a2s):
                    if len(samples) < 100 and randint(1, 10) == 7:
                        logging.info("---------* Sample Demo *------")
                        logging.info("\nA: {}\nP: {}\nN: {}".format(sent, a1, a2))

                    encodings = tokenizer([sent, a1, a2], 
                                truncation=True, padding='max_length', return_tensors='pt')
                    
                    samples.append(encodings)

            pbar.update(1)
        pbar.close() 
        
        return TrainDataset(samples)

    @staticmethod
    def prepare_eval_data(df, tokenizer):
        """
        create test pairs (S1, S2)
        """
        pbar = tqdm(total=len(df))
        samples = {}
        for _, row in df.iterrows():
            if pd.isna(row["sim"]):  # incorrect
                assert not pd.isna(row["alt 2"]) and not pd.isna(row["otherID"])

                samples[row["ID"]] = (row["sent"], row["alt 2"])
                samples[row["otherID"]] = (row["alt 1"], row["alt 2"])
            else:  # correct
                assert pd.isna(row["alt 2"])

                samples[row["ID"]] = (row["sent"], row["alt 1"])

            pbar.update(1)
        pbar.close() 

        ids = list(samples.keys())
        p1s, p2s = [samples[i][0] for i in ids], [samples[i][1] for i in ids]

        logging.info("Data size {}".format(len(samples)))
    
        encodings1 = tokenizer(p1s, truncation=True, padding=True)
        encodings2 = tokenizer(p2s, truncation=True, padding=True)

        return TestDataset(encodings1, encodings2, ids)

    def load_data(self, subset="train"):
        hp = self.args

        logging.info("** -- Dataset: {}".format(subset))

        df = pd.read_json(hp.RAW_DATA_PATH, lines=True)
        df = df[df["set"] == subset]

        cache_pth = Path(hp.OUT_DIR).joinpath("{}-cache-{}".format(Path(hp.pretrained_model_path).stem, subset))
        if cache_pth.exists():
            logging.info("Loading cache from {}".format(cache_pth.as_posix()))
            return torch.load(cache_pth.as_posix()), df

        tokenizer = AutoTokenizer.from_pretrained(hp.pretrained_model_path, use_fast=True)
        tokenizer.model_max_length = hp.MAXLEN

        func = self.prepare_train_data if subset == "train" else self.prepare_eval_data 
        dataset = func(df, tokenizer)
   
        logging.info("Saving cache to {}".format(cache_pth.as_posix()))
        torch.save(dataset, cache_pth.as_posix())

        return dataset, df


def scorer( pred, gt):
    """
    pred: dataframe or dict?
    gt: pandas dataframe

    all ids must be found in pred
    """
    # verification
    pd_ids = list(pred.keys())
    gt_ids = gt["ID"].dropna().drop_duplicates().tolist() + gt["otherID"].dropna().drop_duplicates().tolist()

    if not set(pd_ids).issubset(gt_ids):
        logging.error("!! ID doesnt match.")
        exit()


    gold_labels_all = list()
    predictions_all = list()

    gold_labels_sts = list()
    predictions_sts = list()

    gold_labels_no_sts = list()
    predictions_no_sts = list()

    for _, row in gt.iterrows():
        this_sim = row["sim"]
        id1 = row["ID"]
        try:
            if pd.isna(this_sim):
                id2 = row["otherID"]
                this_sim = pred[id2]

            this_sim = float( this_sim )
            this_prediction = round(float( pred[id1] ), 2)
        except Exception as err:
            print("******", err)
            break

        if pd.isna(row["idiom"]):  # standard sts
            gold_labels_sts.append( this_sim        )
            predictions_sts.append( this_prediction )
        else :
            gold_labels_no_sts.append( this_sim        )
            predictions_no_sts.append( this_prediction )

        gold_labels_all.append( this_sim        )
        predictions_all.append( this_prediction )

    corel_all, pvalue    =  spearmanr(gold_labels_all   , predictions_all    )
    corel_sts, pvalue    =  spearmanr(gold_labels_sts   , predictions_sts    )
    corel_no_sts, pvalue =  spearmanr(gold_labels_no_sts, predictions_no_sts )
    return ( corel_all, corel_sts, corel_no_sts )
   

def preprocess(fpth, fout):
    """
    create pseudo-dataset
    """
    df = pd.read_json(fpth, lines=True)

    subset = df["set"].drop_duplicates().tolist()

    # set_tps, samples1, samples2, labels = [], [],[],[]  # label: 1, 0
    # pos_label = 1
    samples = []
    for each in subset:
        print("--------- current set: {}".format(each))
        this_st = df[df["set"] == each]
        for _, row in this_st.iterrows():

            if pd.isna(row["sim"]):  # incorrect, ists
                assert not pd.isna(row["alt 2"])
                assert not pd.isna(row["otherID"])

                samples.append(
                    {
                    "id": row["ID"],
                    "sent1": row["sent"],
                    "sent2": row["alt 2"],
                    "label": 0,
                    "set": row["set"]
                })

                samples.append(
                    {
                    "id": row["otherID"],
                    "sent1": row["alt 1"],
                    "sent2": row["alt 2"],
                    "label": 0,
                    "set": row["set"]
                })
                sid = row["ID"].split("a2")[0]
                a1id = row["otherID"].split("a2")[0]

                samples.append(
                    {
                    "id": f"{sid}{a1id}",
                    "sent1": row["sent"],
                    "sent2": row["alt 1"],
                    "label": 1,
                    "set": row["set"]
                })

            else:
                assert pd.isna(row["alt 2"])

                {
                    "id": row["ID"],
                    "sent1": row["sent"],
                    "sent2": row["alt 1"],
                    "label": row["sim"],
                    "set": row["set"]
                }


    ndf = pd.DataFrame(samples)
    print(ndf.shape)
    ndf.drop_duplicates(subset="id", inplace=True)
    print(ndf.shape)
    ndf.to_csv(fout, sep="\t", index=False)


if __name__ == "__main__":

    # preprocess(
    #     "/disk/lani/mwe/sts/data/newID_noDuplicates2.json",
    #     "/disk/lani/mwe/sts/data/simcse.csv",
    # )
        

    from hyperparameters import Hyperparameters as hp
    load_data(hp)



