import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from stream import BIN_Data_Encoder

class Cus_DataLoader:
    def __init__(self, dataloader1, dataset, model, device, confidence, ratio, batch_size):
        self.dataloader1 = dataloader1
        
        self.label_data(dataloader1, dataset, model, device, confidence, ratio, batch_size)
        
        self.len_labeled = len(self.dataloader1)
        self.len_unlabeled = len(self.dataloader2)
        self.iter_unlabeled = iter(self.dataloader2)
        self.idx_labeled = 0
        self.idx_unlabeled = 0
        self.iter_count = 0
    
    def label_data(self, dataloader, dataset, model, device, confidence, ratio, batch_size):
        print("Labeling unlabeled data...")
        
        # label with unaugmented unlabeld dataset
        data_size = len(dataset)
        dataset_encoded = BIN_Data_Encoder(dataset.index.values, dataset.Label.values, dataset)
        dataloader = DataLoader(dataset_encoded, batch_size=batch_size, shuffle=False, drop_last=True)
        model.to(device)
        model.eval()
        count = [0] * 2
        table1 = []
        table0 = []
        for idx, (d, p, d_mask, p_mask, label) in enumerate(dataloader):
          with torch.no_grad():
            logit = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())
            m = torch.nn.Sigmoid()
            score = torch.squeeze(m(logit)).cpu()
            
            for i, score in enumerate(score):
              if score >= 0.5:
                if score >= float(confidence):
                  table1.append([idx*batch_size + i, 1, score])
                  count[1] += 1
              else:
                if 1-score >= float(confidence):
                  table0.append([idx*batch_size + i, 0, score])
                  count[0] += 1
        
        print(count)
        count_dif = count[1]-count[0]
        
        if abs(count_dif) > 0.1*(count[0]+count[1]):
          if count_dif>0:
            table1 =sorted(table1, key=lambda x:x[2],reverse=True)
            table1=table1[:count[0]]
          else:
            table0 =sorted(table0, key=lambda x:x[2],reverse=False)
            table0=table0[:count[1]]         
        
        self.table = []
        for t in table1:
          self.table.append(t[0:2])
        for t in table0:
          self.table.append(t[0:2])

        random.seed(42)
        random.shuffle(self.table)
        
        training_set = BIN_Data_Encoder(self.table, self.table, dataset, option=False)
        self.dataloader2 = DataLoader(training_set, batch_size=int(batch_size*ratio), shuffle=True,drop_last=True)
        
        model.cpu()
        del model
        torch.cuda.empty_cache()
        
        print("Labeling completed")
    
    def __len__(self):
        return self.len_labeled
    
    def __iter__(self):
        self.idx_labeled = 0
        self.iter_count = 0
        self.iter_labeled = iter(self.dataloader1)
        return self 
    
    def __next__(self):
        # if self.idx_labeled >= self.len_labeled:
        #     raise StopIteration
        if self.iter_count >= 5 :
          raise StopIteration
        
        if self.idx_labeled >= self.len_labeled:
          self.idx_labeled = 0
          self.iter_labeled = iter(self.dataloader1)
          self.iter_count += 1
          
        d1, p1, d_mask1, p_mask1, label1 = next(self.iter_labeled)
       
        if self.idx_unlabeled >= self.len_unlabeled:
            self.idx_unlabeled = 0
            self.iter_unlabeled = iter(self.dataloader2)
        d2, p2, d_mask2, p_mask2, label2 = next(self.iter_unlabeled)
        
        d = torch.cat((d1, d2), dim=0)
        p = torch.cat((p1, p2), dim=0)
        d_mask = torch.cat((d_mask1, d_mask2), dim=0)
        p_mask = torch.cat((p_mask1, p_mask2), dim=0)
        label = torch.cat((label1, label2), dim=0)
            
        self.idx_labeled += 1
        self.idx_unlabeled += 1
        
        return d, p, d_mask, p_mask, label