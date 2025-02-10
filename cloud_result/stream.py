import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE #서브워드 분할 함수, 
import codecs

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path) #encoding default -> windows cp1252 , linux UTF-8
pbpe = BPE(bpe_codes_protein, merges=-1, separator='') # bpe rule을 학습시키는 과정은 들어가 있지 않음 , merges=-1 모든 병합규칙을 사용한다는 의미, separator=' '해야 이후 .split()제대로 작동할 것 같은데 
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values # protein sub-sequence(word)
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p)))) #dict((key,value)), zip-> tuple, key(sub_word)-> value(index)

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205
max_p = 545
 

def protein2emb_encoder(x):
    max_p = 545 #576
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l)) # real value와 padding 구분하기 위한 input mask
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def drug2emb_encoder(x):
    max_d = 50 #64
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(Dataset):

    def __init__(self, list_IDs, labels, df_dti,option=True):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.option=option
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        if self.option:
          index2 = self.list_IDs[index]
        else:
          index2 = self.list_IDs[index][0]
        
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index2]['SMILES']
        p = self.df.iloc[index2]['Target Sequence']
        
        #d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        if self.option:
          y = self.labels[index2]
        else:
          y = self.labels[index][1]
        return d_v, p_v, input_mask_d, input_mask_p, y