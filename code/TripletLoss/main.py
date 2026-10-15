#!/usr/bin/env python
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path 

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedModel, TrainingArguments
from sentence_transformers    import SentenceTransformer

from sklearn.metrics.pairwise import paired_cosine_distances

import argparse

#! Core!!!
from utils.data_loader import DataProcessor
from utils.utils import scorer
from scr.scr import MyTrainer

import sys
for each in Path.cwd().parents:
    pth = each.as_posix()
    if "pie" in pth:
        sys.path.insert(0, pth)

from common.io_utils import *
from common.setup import logging

# logging.disabled(logging.WARNING)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='tripletloss')
    # model and tokenizer
    parser.add_argument('--model_name', 
                        default="hellonlp/simcse-roberta-large-zh",
                        help='model path')
    parser.add_argument('--model_dir', 
                        default="mwe/sts/triLoss2/all_layer/checkpoint-4968",
                        help='model path for eval')

    parser.add_argument('--max_length', default=128, type=int)
    # dir
    parser.add_argument('--raw_dir', 
                        default="mwe/sts/data/samples2.csv",
                        type=str, 
                        # required=True, 
                        help='raw data path')  
    parser.add_argument('--out_dir', 
                            default="mwe/sts/triLoss2",
                        type=str, 
                        # required=True, 
                        help='output dir') 
    parser.add_argument('--job_name', 
                        default="all_layer",
                        type=str,   
                        help='job name')  # eval, data & model

    parser.add_argument('--label_pth',  
                        default="/disk/lani/mwe/sts/data/newID_noDuplicates2.json")

    # Train config
    parser.add_argument('--agg_mode', help="{tokenmarker4layer|tokenmarker2layer|last_hidden|all_layer}",
                        default="all_layer")
    parser.add_argument('--using_ext_miner',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=1e-16, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=64, type=int)
    parser.add_argument('--eval_batch_size',
                        help='eval batch size',
                        default=64, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=24, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000)
    
    parser.add_argument('--train',  action="store_true")
    parser.add_argument('--parallel', default=True,
                        action="store_true") 
    parser.add_argument('--disable_prog_bar', action="store_true", help="disable progress bar") 
    parser.add_argument('--random_seed',
                        default=22, 
                        type=int)
    parser.add_argument('--loss',
                        help="{ms_loss|cosine_loss|circle_loss|triplet_loss|infoNCE}}",
                        default="triplet_loss")
    # parser.add_argument('--training_mode',
    #                     help="{pre_training|fine_tuning}}",
    #                     default="ms_loss")
    parser.add_argument('--infoNCE_tau', default=0.04, type=float) 
    parser.add_argument('--use_miner', action="store_true") 
    parser.add_argument('--miner_margin', default=0.4, type=float) 
    parser.add_argument('--triplet_margin', default= 0.3, type=float, 
                        help='The desired difference between the pos and neg distance')
    parser.add_argument('--type_of_triplets', default="all", type=str) 
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--is_shuffle', action="store_true")
    parser.add_argument('--pad_in_eos', action="store_true")
    parser.add_argument('--no_progress_bar', action="store_true")
    parser.add_argument('--mean_layer_nums', help='choose numbers of layer to mean', default=4, type=int)    

    return parser.parse_args()

def main(args):

    # set random seeds
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    #? prepare for output
    # out_dir, job_dir: model, eval
    out_dir = Path(args.out_dir).joinpath(args.job_name)
    out_dir.mkdir(exist_ok=True, parents=True)

    #! config
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=torch.cuda.is_available()  ,
        loss=args.loss,
        infoNCE_tau=args.infoNCE_tau,
        use_miner=args.use_miner,
        miner_margin=args.miner_margin,
        triplet_margin=args.triplet_margin,
        type_of_triplets=args.type_of_triplets,
        output_hidden_states=True,
        hidden_dropout_prob=args.dropout_rate,
        attention_probs_dropout_prob=args.dropout_rate
        )

    if args.train:
        logging.info("Loading model from {}".format(args.model_name))
        model = AutoModel.from_pretrained(args.model_name, config=config)  #! automodel or sentence transformer

        logging.info("Loading data ... ")
        train_set = DataProcessor(args).load_data()
        logging.info("** Trainset size: {}".format(len(train_set)))
        # eval_set = torch.load(args.eval_dir)
        training_args = TrainingArguments(
            remove_unused_columns=False, 
            output_dir=out_dir,
            #* training 
            per_device_train_batch_size = args.train_batch_size,
            num_train_epochs=args.epoch,
            save_on_each_node=True,
            save_strategy ='epoch',
            # save_steps =int(len(train_set)/args.train_batch_size),
            learning_rate=args.learning_rate,
            fp16=torch.cuda.is_available(),  # fp16 is true requires cuda
            report_to="none",
            logging_strategy ="epoch",
            disable_tqdm=args.disable_prog_bar,
            local_rank=1,
            #* evaluation todo
            )

        # --------- training
        logging.info("Training ... ")
        trainer =  MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            hp=args,  # hyperparameters
            )
        
        trainer.train()

    else: # evaluation
        
        """inference/ evaluation on cloze test"""
        logging.info("--------* Evaluation *--------")
        logging.info("Loading model from {}".format(args.model_dir))

        eval_save_pth = out_dir.joinpath("{}-eval".format(Path(args.model_dir).stem))

        if eval_save_pth.exists():  # prediction results
            scores = load_json(eval_save_pth)
            _, gts = DataProcessor(args).load_eval_data(subset="test")
            
        else:
            model = AutoModel.from_pretrained(args.model_dir)
            training_args = torch.load(Path(args.model_dir).joinpath("training_args.bin"))

            # eval_set = DataProcessor(args).load_gts(subset="test")
            # todo load different args
            # 这里有两个args，一个是当前的args，一个是上一次保留的训练的args
            eval_set, gts = DataProcessor(args).load_eval_data(subset="test")
            data_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.eval_batch_size)


            model.eval()
            device = "cuda"
            model.to(device)
            scores = {}
            # save scores
            with torch.no_grad():
                for batch in tqdm(data_loader):
                    ids = batch.pop("ids")
                    inputs1 = batch.pop("s1")
                    inputs2 = batch.pop("s2")

                    inputs1 = {key: val.to(device) for key, val in inputs1.items()}
                    inputs2 = {key: val.to(device) for key, val in inputs2.items()}

                    outputs1 = model(**inputs1)
                    outputs2 = model(**inputs2)

                    embs1 = outputs1.last_hidden_state[:, 0].squeeze(0).cpu()
                    embs2 = outputs2.last_hidden_state[:, 0].squeeze(0).cpu()

                    this_score = 1 - paired_cosine_distances(embs1, embs2)
    
                    this_score = list(map(str, np.round(this_score, 6)))

                    scores.update(dict(zip(ids, this_score)))
            dump_json(scores, eval_save_pth)

        corel_all, corel_sts, corel_no_sts = scorer(scores, gts)
        logging.info("Overall: {}".format(corel_all))
        logging.info("STS only: {}".format(corel_sts))
        logging.info("iSTS only: {}".format(corel_no_sts))
    
if __name__ == '__main__':
    main(parse_args())