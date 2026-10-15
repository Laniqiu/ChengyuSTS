import torch
from pytorch_metric_learning.losses import TripletMarginLoss, NPairsLoss#, NTXentLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from transformers import  Trainer
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import normalize
from torch.nn import functional as F


from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence


class MyTrainer(Trainer):
    def __init__(self, *args, hp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hp = hp

        self.loss_cust = TripletMarginLoss(margin=self.hp.triplet_margin, distance=CosineSimilarity())
        self.miner_cust=TripletMarginMiner(margin=self.hp.miner_margin, type_of_triplets=self.hp.type_of_triplets, distance=CosineSimilarity()) #todo distance修改？

    
    def _multi_loss(self, model, inputs, **kwargs):

        inputs1 = inputs.pop("inputs1")
        inputs2 = inputs.pop("inputs2")


        loss1 = self._single_task(model, inputs1)
        loss2 = self._single_task(model, inputs2)

        return loss1 + loss2


    
    def _single_task(self, model, inputs, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        if self.hp.agg_mode =='tokenmarker4layer':
            hidden_state1 = sum(hidden_states[-4:])/4    
        elif self.hp.agg_mode =='tokenmarker2layer':
            hidden_state1 = sum(hidden_states[-2:])/2
        elif self.hp.agg_mode =='all_layer':
            hidden_state1 = sum(hidden_states[:])/len(hidden_states)
        elif self.hp.agg_mode=='last_hidden':
            hidden_state1=outputs.last_hidden_state

        query_embed = hidden_state1.mean(1)  # query : [batch_size, embed_size]

        if self.hp.miner_margin > 0:
            hard_pairs = self.miner_cust(query_embed, labels)
            return self.loss_cust(query_embed, labels, hard_pairs)
        else:
            return self.loss_cust(query_embed, labels)


     
    def compute_loss(self, model, inputs, **kwargs):

        return self._single_task(model, inputs, **kwargs)


        # labels = inputs.pop("labels")

        # outputs = model(**inputs)
        # hidden_states = outputs.hidden_states

        # if self.hp.agg_mode =='tokenmarker4layer':
        #     hidden_state1 = sum(hidden_states[-4:])/4    
        # elif self.hp.agg_mode =='tokenmarker2layer':
        #     hidden_state1 = sum(hidden_states[-2:])/2
        # elif self.hp.agg_mode =='all_layer':
        #     hidden_state1 = sum(hidden_states[:])/len(hidden_states)
        # elif self.hp.agg_mode=='last_hidden':
        #     hidden_state1=outputs.last_hidden_state

        # query_embed = hidden_state1.mean(1)  # query : [batch_size, embed_size]

        # if self.hp.miner_margin > 0:
        #     hard_pairs = self.miner_cust(query_embed, labels)
        #     return self.loss_cust(query_embed, labels, hard_pairs)
        # else:
        #     return self.loss_cust(query_embed, labels)

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        !!!Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset not found!")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle":False
        }
        return DataLoader(train_dataset, **dataloader_params)



    
    




