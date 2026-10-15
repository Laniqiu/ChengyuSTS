import csv
from scipy.stats import pearsonr, spearmanr
import pandas as pd

def scorer( pred, gt):
    """
    pred: dataframe or dict?
    gt: pandas dataframe
    """

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
   