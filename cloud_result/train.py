# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + executionInfo={"elapsed": 553, "status": "ok", "timestamp": 1734524820418, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="U9aQ_JEMBQjA"
import os
#os.chdir('/content/drive/MyDrive/mini')

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3471, "status": "ok", "timestamp": 1734524824358, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="l1PBvnN3BbQG" outputId="5278e3cd-ce78-4f0a-b38f-975a09910d74"

# + executionInfo={"elapsed": 17214, "status": "ok", "timestamp": 1734524841568, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="N0Gg5-8UkbNV"
import copy
from time import time
import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import collections
import math
import copy
from subword_nmt.apply_bpe import BPE
import codecs

torch.manual_seed(2)
np.random.seed(3)

from model import BIN_Interaction_Flat1, BIN_Interaction_Flat2, BIN_Interaction_Flat3
from stream import BIN_Data_Encoder
from data import Cus_DataLoader


# + executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1734524841568, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="NDO4E5H6kDzY"
def test(data_generator, model):
  torch.cuda.empty_cache()
  y_pred = []
  y_label = []
  model.eval()
  loss_accumulate = 0.0
  count = 0.0
  for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
    score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda()) # long tensor int64, .to(device)를 왜 사용안했지?

    m = torch.nn.Sigmoid()
    logits = torch.squeeze(m(score)) # 차원이 1인 축 제거
    loss_fct = torch.nn.BCELoss()

    label = label.float().cuda()

    loss = loss_fct(logits, label)

    loss_accumulate += loss.item()
    count += 1

    logits = logits.detach().cpu().numpy()

    label_ids = label.detach().to('cpu').numpy()
    y_label = y_label + label_ids.flatten().tolist()
    y_pred = y_pred + logits.flatten().tolist()

  loss = loss_accumulate / count

  fpr, tpr, thresholds = roc_curve(y_label, y_pred) # TPR: 양성을 잘 잡아내는 능력 값이 클수록 재현율이 높음, FPR: 음성을 얼마나 잘 음성으로 유지하는지 값이 작을수록 좋으며 음성을 양성으로 잘못 분류하는 비율이 낮음을 의미

  precision = tpr / (tpr + fpr+0.00001)

  f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

  thred_optim = thresholds[2:][np.argmax(f1[2:])] # threshold는 y_pred에 대해 내림차순으로 뱉어내므로 너무 strict한 threshold 5개는 고려하지않음 그 이유는?

  print("optimal threshold: " + str(thred_optim))

  y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

  auc_k = auc(fpr, tpr)
  print("AUROC:" + str(auc_k))
  print("AUPRC: " + str(average_precision_score(y_label, y_pred))) # 여러 threshold 기준 recall-precision auc

  cm1 = confusion_matrix(y_label, y_pred_s)
  print('Confusion Matrix : \n', cm1)
  print('Recall : ', recall_score(y_label, y_pred_s))
  print('Precision : ', precision_score(y_label, y_pred_s)) # 단일 threshold 기준(최적값)

  total1 = sum(sum(cm1))
  #####from confusion matrix calculate accuracy
  accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
  print('Accuracy : ', accuracy1)

  sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
  print('Sensitivity : ', sensitivity1)

  specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
  print('Specificity : ', specificity1)

  outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
  return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), y_pred, loss

def train(model,training_generator,epochs,batchsize,lr):
  torch.cuda.empty_cache()
  loss_history = []
  #model = model.cuda()

  opt = torch.optim.Adam(model.parameters(), lr=lr)
  print('--- Data Preparation ---')
  params = {'batch_size': batchsize, 'shuffle': True, 'drop_last': True}

  dataFolder = os.getcwd()
  df_val = pd.read_csv(dataFolder + '/val.csv') #valid_new / test_new
  df_test = pd.read_csv(dataFolder + '/test.csv')

  validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
  validation_generator = DataLoader(validation_set, **params)

  testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
  testing_generator = DataLoader(testing_set, **params)

  # early stopping
  max_auc = 0
  model_max = copy.deepcopy(model.to('cpu'))
  epo_count = 0

  with torch.no_grad(): # with torch.no_grad():
    model_max = model_max.cuda()
    auc, auprc, f1, logits, loss = test(testing_generator, model_max)
    print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(loss))
    model_max = model_max.cpu()
    torch.cuda.empty_cache()

  model=model.cuda()
  print('--- Go for Training ---')
  for epo in range(epochs):
    model.train()
    count=0.0
    loss_accumulate = 0.0
    for i,(d, p, d_mask, p_mask, label) in tqdm(enumerate(training_generator),desc="training"):
      score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

      label = label.float().cuda()

      loss_fct = torch.nn.BCELoss()
      m = torch.nn.Sigmoid()
      n = torch.squeeze(m(score))

      loss = loss_fct(n, label)

      loss_accumulate += loss.cpu().detach().numpy().item()
      count += 1

      opt.zero_grad()
      loss.backward()
      opt.step()

    loss = loss_accumulate / count
    loss_history.append(loss)
    torch.cuda.empty_cache()
    if ((epo+1) % 10 == 0):
      print('Training at Epoch ' + str(epo + 1) + ' with loss ' + str(loss))

    # every epoch test
    with torch.set_grad_enabled(False):
      auc, auprc, f1, logits, loss = test(validation_generator, model)
      if auc > max_auc:
        model_max = copy.deepcopy(model.to('cpu'))
        model=model.cuda()
        epo_count = 0
        max_auc = auc
        print('--------'+str(epo+1)+'model_max updated-------')
      else:
        epo_count += 1
      print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Validation loss: ' + str(loss))
      torch.cuda.empty_cache()
      if epo_count == 20 :
        break
  model.cpu()
  del model
  torch.cuda.empty_cache()
  print('--- Go for Testing ---')
  try:
    with torch.set_grad_enabled(False):
      model_max=model_max.cuda()
      auc, auprc, f1, logits, loss = test(testing_generator, model_max)
      print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(loss))
      model_max=model_max.cpu()
      torch.cuda.empty_cache()
  except:
      print('testing failed')
  return model_max, loss_history


# + executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1734524841568, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="XqXq-Y6b5Tl8"
def BIN_config_DBPE():
    config = {}
    config['batch_size'] = [4,32,64]
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['max_drug_seq'] = 50 #64
    config['max_protein_seq'] = 545 #576
    config['emb_size'] = 384
    config['dropout_rate'] = [0.1,0.15,0.2]

    #DenseNet
    # config['scale_down_ratio'] = 0.25
    # config['growth_rate'] = 20
    # config['transition_rate'] = 0.5
    # config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3

    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['attention_probs_dropout_prob'] = [0.1,0.15,0.2]
    config['hidden_dropout_prob'] = [0.1,0.15,0.2]
    config['flat_dim'] = 78192 #106764
    return config


# + executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1734524841568, "user": {"displayName": "\ucd5c\ud604\uc601", "userId": "12280114103183972011"}, "user_tz": -540} id="tWaLIOo7uuj6"
def main():
  batchsize = 4 # 16
  epochs =300 # 50
  #epochs_fine=50
  lr1 = 5e-5
  lr = 1e-4 #5e-5
  confidence=0.55
  ratio1=7
  ratio2=15
  config = BIN_config_DBPE()
  model = BIN_Interaction_Flat1(**config)
  model2 = BIN_Interaction_Flat2(**config)
  model3 = BIN_Interaction_Flat3(**config)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")

  dataFolder = os.getcwd()
  df_train = pd.read_csv(dataFolder + '/train.csv') #train_new
  df_unlabel = pd.read_csv(dataFolder + '/result.csv')

  params = {'batch_size': batchsize, 'shuffle': True, 'drop_last': True}
  params2 = {'batch_size': batchsize*16, 'shuffle': True, 'drop_last': True}
  training_set = BIN_Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
  training_generator = DataLoader(training_set, **params)
  training_generator2 = DataLoader(training_set, **params2)

  #phase1
  print("-------phase1-------")
  model_max, loss_history = train(model,training_generator,epochs,batchsize,lr1)
  torch.save(model_max, 'model_max.pt') # 임시추가
  loss1= pd.Series(loss_history)
  loss1.to_csv('loss1.csv',index=False)
  torch.cuda.empty_cache()

  #phase2,3,4
  print("-------phase2--------")
  training_generator_with_unlabel=Cus_DataLoader(training_generator, df_unlabel, model_max, device, confidence, ratio1, batchsize)
  torch.cuda.empty_cache()
  model_max2, loss_history2 = train(model2,training_generator_with_unlabel,epochs*2,batchsize*8,lr)
  torch.save(model_max2, 'model_max2.pt') # 임시추가
  loss2= pd.Series(loss_history2)
  loss2.to_csv('loss2.csv',index=False)
  torch.cuda.empty_cache()
  # print("-------phase2 finetuning-------")
  # model_max2, loss_history2 = train(model_max2,training_generator2,epochs_fine*2,batchsize*4, lr)
  # torch.cuda.empty_cache()

  print("-------phase3--------")
  model_max2.batch_size=batchsize
  training_generator_with_unlabel=Cus_DataLoader(training_generator, df_unlabel, model_max2, device, confidence, ratio1, batchsize)
  torch.cuda.empty_cache()
  model_max3, loss_history3 = train(model2,training_generator_with_unlabel,epochs*2,batchsize*8,lr)
  torch.save(model_max3, 'model_max3.pt') # 임시추가
  loss3= pd.Series(loss_history3)
  loss3.to_csv('loss3.csv',index=False)
  torch.cuda.empty_cache()

  print("-------phase4--------")
  model_max3.batch_size=batchsize
  training_generator_with_unlabel=Cus_DataLoader(training_generator, df_unlabel, model_max3, device, confidence, ratio2, batchsize)
  torch.cuda.empty_cache()
  model_max4, loss_history4 = train(model3,training_generator_with_unlabel,epochs*2,batchsize*16,lr)
  torch.save(model_max4, 'model_max4.pt') # 임시추가
  loss4= pd.Series(loss_history4)
  loss4.to_csv('loss4.csv',index=False)
  torch.cuda.empty_cache()
  # print("-------phase3 finetuning-------")
  # model_max3, loss_history3 = train(model_max3,training_generator2,epochs_fine*3,batchsize*4,lr)
  # torch.cuda.empty_cache()

  print("-------phase5--------")
  model3 = BIN_Interaction_Flat3(**config)
  model_comparison, loss_history_comparison = train(model3, training_generator2, epochs*2, batchsize*16, lr)
  torch.save(model_comparison, 'model_comparison.pt') # 임시추가
  loss_comparison= pd.Series(loss_history_comparison)
  loss_comparison.to_csv('loss_comparison.csv',index=False)
  torch.cuda.empty_cache()

  print("-------training finished-------")

  return model_max, model_max2, model_max3, model_comparison, loss_history, loss_history2, loss_history3, loss_history_comparison

# + colab={"base_uri": "https://localhost:8080/"} id="4AMfsnpIupjf" outputId="0bef5d4f-4ea0-4f65-c677-11b619442f81"
s = time()
model_max, model_max2, model_max3, model_comparison, loss_history, loss_history2, loss_history3, loss_history_comparison = main()
e = time()
print((e - s)/60)

# torch.save(model_max, 'model_max.pt')
# torch.save(model_max2, 'model_max2.pt')
# torch.save(model_max3, 'model_max3.pt')
# torch.save(model_comparison, 'model_comparison.pt')
# loss= pd.Series(loss_history+loss_history2+loss_history3+loss_history_comparison)
# loss.to_csv('loss.csv',index=False)
