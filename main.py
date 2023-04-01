import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any, Tuple
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.nn.functional as F
import os
from transformers import BartForConditionalGeneration, TrainingArguments, T5ForConditionalGeneration
import torch.nn.functional as F
import json
import torch
from torch import nn
#import tensorflow as tf
from torch.utils.data import Dataset
from pprint import pprint
from tqdm import tqdm
import torch.nn.functional as F
import random
import re
import itertools
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import json
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
#import sklearn
#from sklearn.metrics import precision_score as sk_precision
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
#from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import huggingface_hub
import spacy
from transformers import BartTokenizer, BartModel, T5Tokenizer 
from pytorch_lightning.loggers import WandbLogger
from encoder_sentence import encode_sentence_Batch,encode_sentence,tokenizer
from batches_management import get_batches_dataset_division
from Model import MLP_arg_classification
from data_extractor import ReaderM2
from Trainer import Model_Correction
import argparse
import MyDataLoader
from MyDataLoader import MyDataModule

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device ="cuda"
#***---> READ M2***
nlp = spacy.load('it_core_news_sm')
max_len=10
classes_ERRANT={}
torch.cuda.empty_cache()

class Main:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-l",action="store_true", help="Use Leonide")
        parser.add_argument("-b",action="store_true", help="per batch division")
        parser.add_argument("-a",default='train',type=str, help="Action between train/test/dev")
        parser.add_argument("--num_tokens",default=256,type=int, help="number of tokens max for each batch")
        parser.add_argument("-v",action="store_true", help="Verbose print of all txt outputs")
        #parser.add_argument("--help",action="store_true", help="shows options")
        self.args = parser.parse_args()
        self.leonide = self.args.l
        self.batch = self.args.b
        self.todo = self.args.a
        self.verbose = self.args.v
        self.num_tokens = self.args.num_tokens
    def run(self):
        print(f"The value you passed is leonide={self.leonide},batch={self.batch}")
        ###### M2 Reader ######
        # MERLIN
        """
        #DataLoader used for train/test/dev on Merlin
        data1_1 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_train.m2',
                 '/media/errant_env/errant/MERLIN/Merlin/dataset/train.txt',classes_ERRANT)
        data1_2 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_test.m2',
                   '/media/errant_env/errant/MERLIN/Merlin/dataset/test.txt',classes_ERRANT)
        data1_3 = ReaderM2('/media/errant_env/errant/MERLIN/Merlin/dataset/original_dev.m2',
                 '/media/errant_env/errant/MERLIN/Merlin/dataset/dev.txt',classes_ERRANT)
        torch.save(data1_1,'/media/data/train.pt')
        torch.save(data1_2,'/media/data/test.pt')
        torch.save(data1_3,'/media/data/dev.pt')
        
        ###### LEONIDE ######
        
        #DataLoader used for train/test/dev on Leonide
        data1_1L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_train.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_train.txt',classes_ERRANT)
        data1_2L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_test.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_test.txt',classes_ERRANT)
        data1_3L = ReaderM2('/media/errant_env/errant/LEONIDE/Leonide/original_dev.m2',
                   '/media/errant_env/errant/LEONIDE/Leonide/original_dev.txt',classes_ERRANT)
        torch.save(data1_1L,'/media/data/trainL.pt')
        torch.save(data1_2L,'/media/data/testL.pt')
        torch.save(data1_3L,'/media/data/devL.pt')
        """

        """#***---> Loads***"""
        data1_1 =torch.load('/media/data/train.pt')
        data1_1.Y_list=[]
        data1_2 =torch.load('/media/data/test.pt')
        data1_2.Y_list=[]
        data1_3 =torch.load('/media/data/dev.pt')
        data1_3.Y_list=[]

        """#***---> Loads***"""
        data1_1L =torch.load('/media/data/trainL.pt')
        data1_1L.Y_list=[]
        data1_2L =torch.load('/media/data/testL.pt')
        data1_2L.Y_list=[]
        data1_3L =torch.load('/media/data/devL.pt')
        data1_3L.Y_list=[]


        # MERLIN
        ##### Tokenization Merlin ####
        if self.batch and not self.leonide:
          ##### batch ####
          print(f"1. NOT Leonide and batches division  ")
          tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1.X_list,data1_1.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch(data1_2.X_list,data1_2.X_corr)
          #############per Batch Division#############
          dataset1,correct1= get_batches_dataset_division(
                    tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],self.num_tokens)
          dataset2,correct2= get_batches_dataset_division(
                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],self.num_tokens)
          dataset3,correct3= get_batches_dataset_division(
                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],self.num_tokens)
          data = MyDataModule(
                dataset1['input_ids'],
                correct1['input_ids'],
                dataset1['attention_mask'],
                correct1['attention_mask'],
                dataset2['input_ids'],
                correct2['input_ids'],
                dataset2['attention_mask'],
                correct2['attention_mask'],
                dataset3['input_ids'],
                correct3['input_ids'],
                dataset3['attention_mask'],
                correct3['attention_mask'],
                True)
        elif not self.leonide:
          print(f"2. NOT Leonide and NOT batches division  ")
          ##### no batch ####
          tokenized_seq1,tokenized_corr = encode_sentence(data1_1.X_list,data1_1.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence(data1_2.X_list,data1_2.X_corr)
          #############Original Batch#############
          data = MyDataModule(tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],

                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],

                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],
                    False)
        
        # MERLIN + LEONIDE
        ##### Tokenization Leonide ####
        if self.batch and self.leonide:
          print(f"3. Use Leonide and batches division ")
          ##### batch ####
          tokenized_seq1,tokenized_corr = encode_sentence_Batch(data1_1.X_list+data1_1L.X_list+data1_2L.X_list+data1_3L.X_list,data1_1.X_corr+data1_1L.X_corr+data1_2L.X_corr+data1_3L.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence_Batch(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence_Batch(data1_2.X_list,data1_2.X_corr)
          #############per Batch Division#############
          dataset1,correct1= get_batches_dataset_division(
                    tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],self.num_tokens)
          dataset2,correct2= get_batches_dataset_division(
                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],self.num_tokens)
          dataset3,correct3= get_batches_dataset_division(
                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],self.num_tokens)
          data = MyDataModule(
                dataset1['input_ids'],
                correct1['input_ids'],
                dataset1['attention_mask'],
                correct1['attention_mask'],
                dataset2['input_ids'],
                correct2['input_ids'],
                dataset2['attention_mask'],
                correct2['attention_mask'],
                dataset3['input_ids'],
                correct3['input_ids'],
                dataset3['attention_mask'],
                correct3['attention_mask'],
                True)
        elif self.leonide:
          print(f"4. Use Leonide but NOT batches division ")
          ##### no batch ####
          tokenized_seq1,tokenized_corr = encode_sentence(data1_1.X_list+data1_1L.X_list+data1_2L.X_list+data1_3L.X_list,data1_1.X_corr+data1_1L.X_corr+data1_2L.X_corr+data1_3L.X_corr)
          tokenized_seq_dev1,tokenized_corr_dev = encode_sentence(data1_3.X_list,data1_3.X_corr)
          tokenized_seq_test1,tokenized_corr_test = encode_sentence(data1_2.X_list,data1_2.X_corr)
          #############Original Batch#############
          data = MyDataModule(tokenized_seq1['input_ids'],
                    tokenized_corr['input_ids'],
                    tokenized_seq1['attention_mask'],
                    tokenized_corr['attention_mask'],
                    tokenized_seq_dev1['input_ids'],
                    tokenized_corr_dev['input_ids'],
                    tokenized_seq_dev1['attention_mask'],
                    tokenized_corr_dev['attention_mask'],
                    tokenized_seq_test1['input_ids'],
                    tokenized_corr_test['input_ids'],
                    tokenized_seq_test1['attention_mask'],
                    tokenized_corr_test['attention_mask'],
                    False)
        torch.cuda.empty_cache()
        if self.todo=="train":
          print("\n\n START TRAINING \n\n")
          model = Model_Correction(False,MLP_arg_classification(),self.verbose)
          trainer = pl.Trainer(max_epochs = 10,logger=model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=0.5)
          trainer.fit(model,train_dataloaders= data)
        elif self.todo=="dev":
          print("\n\n DEV CHECK \n\n")
          model = Model_Correction(True,torch.load('/media/models/backup_epoch.pt'),self.verbose)
          trainer = pl.Trainer(max_epochs = 1,logger=model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=0.5)
          trainer.validate(model, dataloaders=data)
        elif self.todo=="test":
          print("\n\n TEST CHECK \n\n")
          model = Model_Correction(False,torch.load('/media/models/backup_epoch.pt'),self.verbose)
          trainer = pl.Trainer(max_epochs = 1,logger=model.wandb_logger ,accelerator="cuda", devices=1, val_check_interval=0.5)
          trainer.test(model, dataloaders=data)
        else:
           print(f"No existing action for {self.todo}")
        wandb.finish()

if __name__ == '__main__':
    main = Main()
    main.run()