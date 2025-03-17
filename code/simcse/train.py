# -*- coding: utf-8 -*-
"""
"""


import random
import numpy as np

# from loguru import logger
from scipy.stats import spearmanr

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

from hyperparameters import Hyperparameters as hp
from utils import DataProcessor, scorer
from networks import SimCSEModelSup


import logging


    
def simcse_sup_loss(y_pred):
    """
    有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=hp.DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=hp.DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
    

def eval_spearmanr(model, inputs) -> float: # todo re-write
    """模型评估函数 
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    dataloader, gts = inputs

    device = hp.DEVICE

    sim_tensor = torch.tensor([], device=hp.DEVICE)
    pd_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            ids = batch.pop("id")
            inputs1 = batch.pop("s1")
            inputs2 = batch.pop("s2")

            inputs1 = {key: val.to(device) for key, val in inputs1.items()}
            inputs2 = {key: val.to(device) for key, val in inputs2.items()}
            if "token_type_ids" not in inputs1 or "xlnet" in hp.pretrainebred_model_path:
                inputs1["token_type_ids"] = None
                inputs2["token_type_ids"] = None
        
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

            sim = F.cosine_similarity(outputs1, outputs2, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            pd_ids += ids

    scores =  list(map(str, np.round(sim_tensor.tolist(), 6)))
    preds =  dict(zip(pd_ids, scores))

    corel_all, corel_sts, corel_no_sts = scorer(preds, gts)
    logging.info("Overall: {:.4f}".format(corel_all))
    logging.info("STS only: {:.4f}".format(corel_sts))
    logging.info("iSTS only: {:.4f}".format(corel_no_sts))

    return corel_all
        

def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练
    """
    best = 0.2
    for epoch in range(hp.EPOCHS):
        model.train()
        early_stop_batch = 0
        for batch_idx, source in enumerate(tqdm(train_dl, desc="Epoch {}".format(epoch)), start=1):
            # 维度转换 [batch, 3, seq_len] -> [batch * 3, seq_len]
            real_batch_num = source.get('input_ids').shape[0]

            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(hp.DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(hp.DEVICE)

            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(hp.DEVICE) if "token_type_ids" in source and "xlnet" not in hp.pretrainebred_model_path else None
            # 训练
            out = model(input_ids, attention_mask, token_type_ids) 
            loss = simcse_sup_loss(out)
            # logging.info(f'loss: {loss.item():.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # epoch过半时和结束时评估？？
            if batch_idx in [int(len(train_dl)/2), len(train_dl)]:
                logging.info(f'loss: {loss.item():.4f}')
                corrcoef = eval_spearmanr(model, dev_dl)
                model.train()
                if best < corrcoef:
                    early_stop_batch = 0
                    best = corrcoef
                    save_pth = hp.JOB_DIR.joinpath("e{}-b{}-l{:.3g}-s{:.3g}".format(epoch, batch_idx, loss, best)).as_posix()
                    torch.save(model.state_dict(), save_pth)# + str(batch_idx)
                    logging.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                    continue
                early_stop_batch += 1
                if early_stop_batch == 10:
                    logging.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                    logging.info(f"train use sample number: {(batch_idx - 10) * hp.BATCH_SIZE}")
                    return 
    logging.info(f'Trainining done, models saved at {hp.JOB_DIR.as_posix()}')
    
                
def main(args):

    logging.info(f'device: {hp.DEVICE}, pooling: {hp.POOLING}, model path: {hp.pretrained_model_path}')
    assert hp.POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']

    logging.info("Preparing data...")
    DP = DataProcessor(hp)
    train_data, _ = DP.load_data()
    dev_data, dev_gt = DP.load_data("dev")
    test_data, test_gt = DP.load_data("test")

    train_dataloader = DataLoader(train_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=hp.BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=hp.BATCH_SIZE)

    logging.info("Loading model from {}".format(hp.pretrained_model_path))
    model = SimCSEModelSup(pretrained_model=hp.pretrained_model_path, pooling=hp.POOLING)
    model.to(hp.DEVICE)
    
    # train
    if args.train:  
        hp.model_name = ""
        train(model, train_dataloader, (dev_dataloader, dev_gt), 
                optimizer=torch.optim.AdamW(model.parameters(), lr=hp.LR))
    # eval
    else: 
        breakpoint() 
        logging.info("Loading weights from {}".format(hp.WEIGHTS_DIR))
        model.load_state_dict(torch.load(hp.WEIGHTS_DIR))

        dev_corrcoef = eval_spearmanr(model, (dev_dataloader, dev_gt))
        test_corrcoef = eval_spearmanr(model, (test_dataloader, test_gt))

 

if __name__ == '__main__':      
    import argparse

    parser = argparse.ArgumentParser(description='tripletloss')
    parser.add_argument('--train',  action="store_true")

    main(parser.parse_args())



    

