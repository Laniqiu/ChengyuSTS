"""
data pre-processing: basics, errands
"""
from pathlib import Path
import pandas as pd

from tqdm import tqdm

import logging


def preprocess(fpth, fout):
    """create triplet
    sentences and their correct paraphases have same labels and  each incorrect paraphase has different labels

    先分组，每组含句子s， correct repl和incorrect repl
    (s, a1), (s, a2), (a1, a2)
    目的是为了让idiom和他的correct paraphrase在一起
    所有s和a1有相同的lable，每一个a2有不同的label 
        -> 同一个句子和他的s1标签相同， --> 是这一个。。。
    每个group的label也不同
    """
    df = pd.read_json(fpth, lines=True)

    subset = df["set"].drop_duplicates().tolist()

    samples, labels = [], []  # label: 1, 0
    set_tps = []
    pos_label = 1
    for each in subset:
        print("--------- current set: {}".format(each))

        this_st = df[df["set"] == each]
        for sent in this_st["sent"].drop_duplicates():
            this = this_st[this_st["sent"] == sent]

            for idm in this["idiom"].drop_duplicates():
                if not idm:  # std sts
                    a1s = this[this['sim']==1]["alt 1"].drop_duplicates().tolist()
                    a2s = this[this["sim"]==0]["alt 1"].drop_duplicates().tolist()
                else:
                    this = this[this["idiom"]==idm]  #! 是不是错了

                    a1s, a2s = this["alt 1"].drop_duplicates().tolist(), this["alt 2"].drop_duplicates().tolist()

            assert None not in a1s
            if None in a2s:
                a2s.remove(None)
            
            this_samples = [sent] + a1s + a2s
            this_labels = [pos_label] * (len(a1s) + 1) + [i for i in range(pos_label +1, pos_label + len(a2s) +1)]
            this_tps = [each] * len(this_labels)

            assert len(this_samples) == len(this_labels) and len(set(this_labels)) == len(a2s) + 1

            pos_label += len(a2s) + 1

            samples += this_samples
            labels += this_labels
            set_tps += this_tps
    
    ndf = pd.DataFrame()
    ndf["label"] = labels
    ndf["sentence"] = samples
    ndf["set"] = set_tps
    ndf.to_csv(fout, sep="\t")


def preprocess2(fpth, fout):
    """create triplet
    和preprocess不同的是，让idiom的句子都在一起
    sentences and their correct paraphases have same labels and  each incorrect paraphase has different labels

    先分组，每组含句子s， correct repl和incorrect repl
    (s, a1), (s, a2), (a1, a2)
    目的是为了让idiom和他的correct paraphrase在一起
    所有s和a1有相同的lable，每一个a2有不同的label 
        -> 同一个句子s和他的a1标签相同， 而从s衍生的所有a2标签不同
    每个group的label也不同
    """
    df = pd.read_json(fpth, lines=True)

    subset = df["set"].drop_duplicates().tolist()

    samples, labels = [], []  # label: 1, 0
    set_tps = []
    pos_label = 1
    idioms = []
    for each in subset:
        print("--------- current set: {}".format(each))
        this_st = df[df["set"] == each]

        sts = this_st[this_st["idiom"].isna()]
        # sts
        for sent in sts["sent"].drop_duplicates():
            this = sts[sts["sent"] == sent]
            a1s = this[this['sim']==1]["alt 1"].drop_duplicates().tolist()
            a2s = this[this["sim"]==0]["alt 1"].drop_duplicates().tolist()

            # assert None not in a1s and None not in a2s

            this_samples = [sent] + a1s + a2s
            this_labels = [pos_label] * (len(a1s) + 1) + [i for i in range(pos_label +1, pos_label + len(a2s) +1)]
            this_tps = [each] * len(this_labels)
            this_idms = [None] * len(this_labels)

            pos_label += len(a2s) + 1

            samples += this_samples
            labels += this_labels
            set_tps += this_tps
            idioms += this_idms

        idms = this_st[~this_st["idiom"].isna()].idiom.drop_duplicates()
        # ists
        for idm in idms:
            idm_st = this_st[this_st["idiom"] == idm]
            for sent in idm_st["sent"].drop_duplicates():
                this = idm_st[idm_st["sent"] == sent]
                a1s, a2s = this["alt 1"].drop_duplicates().tolist(), this["alt 2"].drop_duplicates().tolist()
                assert None not in a1s
                if None in a2s:
                    a2s.remove(None)
                
                this_samples = [sent] + a1s + a2s
                this_labels = [pos_label] * (len(a1s) + 1) + [i for i in range(pos_label +1, pos_label + len(a2s) +1)]
                this_tps = [each] * len(this_labels)
                this_idms = [idm] * len(this_labels)

                assert len(this_samples) == len(this_labels) and len(set(this_labels)) == len(a2s) + 1

                pos_label += len(a2s) + 1

                samples += this_samples
                labels += this_labels
                set_tps += this_tps
                idioms += this_idms

    
    ndf = pd.DataFrame()
    ndf["idiom"] = idioms
    ndf["label"] = labels
    ndf["sentence"] = samples
    ndf["set"] = set_tps
    ndf.to_csv(fout, sep="\t", index=False)


def preprocess3(fpth, fout):
    """create triplet
    一个idiom的所有s和a1是同一个label，而对应的所有a2是不同的label

    sentences and their correct paraphases have same labels and  each incorrect paraphase has different labels

    先分组，每组含句子s， correct repl和incorrect repl
    (s, a1), (s, a2), (a1, a2)
    目的是为了让idiom和他的correct paraphrase在一起
    所有s和a1有相同的lable，每一个a2有不同的label 
        -> 同一个句子s和他的a1标签相同， 而从s衍生的所有a2标签不同
    每个group的label也不同
    """
    df = pd.read_json(fpth, lines=True)

    subset = df["set"].drop_duplicates().tolist()

    samples, labels = [], []  # label: 1, 0
    set_tps = []
    pos_label = 1
    idioms = []
    for each in subset:
        print("--------- current set: {}".format(each))
        this_st = df[df["set"] == each]

        idms = this_st.idiom.drop_duplicates().tolist()
        for idm in idms:

            idm_st = this_st[this_st["idiom"] == idm] if idm else this_st[this_st["idiom"].isna()]
            for sent in idm_st["sent"].drop_duplicates():
                this = idm_st[idm_st["sent"] == sent]
                if idm:
                    a1s, a2s = this["alt 1"].drop_duplicates().tolist(), this["alt 2"].drop_duplicates().tolist()
                else:
                    a1s = this[this['sim']==1]["alt 1"].drop_duplicates().tolist()
                    a2s = this[this["sim"]==0]["alt 1"].drop_duplicates().tolist()

                assert None not in a1s
                if None in a2s:
                    a2s.remove(None)
                
                this_samples = [sent] + a1s + a2s
                this_labels = [pos_label] * (len(a1s) + 1) + [i for i in range(pos_label +1, pos_label + len(a2s) +1)]
                this_tps = [each] * len(this_labels)

                assert len(this_samples) == len(this_labels) and len(set(this_labels)) == len(a2s) + 1

                pos_label += len(a2s) + 1

                samples += this_samples
                labels += this_labels
                set_tps += this_tps
                idioms += [idm] * len(this_tps)
    
    ndf = pd.DataFrame()
    ndf["idiom"] = idioms
    ndf["label"] = labels
    ndf["sentence"] = samples
    ndf["set"] = set_tps

    ndf.to_csv(fout, sep="\t", index=False)




if __name__ == "__main__":


    """prepare data for training/ evaluation"""

    # out_dir = Path("/mwe/")
    # out_dir.mkdir(exist_ok=True)

    # MODEL_NAME = None

    # processor = DataProcessor(out_dir)
    # # !prepare data > raw
    # todo: + 一个分类任务，同步训练，同一个idiom的所有s和a1认为是一个类别？
    # preprocess(
    #     "/mwe/sts/data/newID_noDuplicates2.json",
    #     "/mwe/sts/data/samples.csv",
    # )

    preprocess2(
        "/mwe/sts/data/newID_noDuplicates2.json",
        "/mwe/sts/data/samples2.csv",
    )


    #  preprocess3(
    #     "/mwe/sts/data/newID_noDuplicates2.json",
    #     "/mwe/sts/data/samples3.csv",
    # )

