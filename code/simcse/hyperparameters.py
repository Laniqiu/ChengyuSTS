# -*- coding: utf-8 -*-
"""
"""


# import os
# import sys
import torch
# pwd = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(pwd)

from pathlib import Path


class Hyperparameters:
    """ 参数设置 """
    
    # 基本参数
    EPOCHS = 24
    SAMPLES = 10000 
    BATCH_SIZE = 64
    LR = 1e-5
    DROPOUT = 0.1 
    MAXLEN = 64
    POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # 预训练模型目录
    # pretrained_model_path =  "hfl/chinese-roberta-wwm-ext"
    # pretrained_model_path =  "hfl/chinese-bert-wwm-ext"
    # pretrained_model_path =  "google-bert/bert-base-chinese"
    # pretrained_model_path = "hfl/chinese-xlnet-base" 
    pretrained_model_path = "FacebookAI/xlm-roberta-base"
    
    # 路径
    RAW_DATA_PATH = Path("newID_noDuplicates2.json")
    OUT_DIR = Path("simcse/")  # jobs using same data saved the same dir
    job_name = Path(pretrained_model_path).stem
    model_name = ""  # for evaluation

    # 路径建设
    JOB_DIR = OUT_DIR.joinpath(job_name)
    JOB_DIR.mkdir(exist_ok=True, parents=True)
    WEIGHTS_DIR = JOB_DIR.joinpath(model_name).as_posix()


    
 

if __name__ == '__main__': 
    #
    hp = Hyperparameters()
    print(hp.pretrained_model_path)
    
