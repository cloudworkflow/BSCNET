#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label_nums',default=7)
parser.add_argument('--context_type',default="P4")
parser.add_argument('--use_uda',default=True)
parser.add_argument('--load_weight',default=False)
parser.add_argument('--save_weights',default=False)
parser.add_argument('--dataset',default=1)
parser.add_argument('--label_size',default=0.9)
#parser.add_argument('--', default=)
opt = parser.parse_args()

if opt.label_nums !=2 or opt.label_nums!=7:
    raise AssertionError('label_nums should be 2/7')
label_nums = opt.label_nums

use_uda = opt.use_uda
load_weight = opt.load_weight
save_weights = opt.save_weights

if opt.dataset == 2 and opt.context_type != 'P4':
    raise AssertionError('Only the combination of P4 is supported on this dataset.')

if opt.context_type == 'P0':
    use_context = False
    context_type = ""
    context_pos = True
elif opt.context_type == 'P1':
    use_context = True
    context_type = "fcon"
    context_pos = True
elif opt.context_type == 'P2':
    use_context = True
    context_type = "zcon"
    context_pos = True
elif opt.context_type == 'P3':
    use_context = True
    context_type = "context"
    context_pos = True
else opt.context_type == 'P4':
    use_context = True
    context_type = "context"
    context_pos = False

if opt.dataset == 1:
    testpath='./data1/test.csv'
    ori_path='./data1/unlabel_data.csv'
    if opt.label_size = 0.9
        trainpath='./data1/0.9train.csv'
    elif opt.label_size = 0.6
        trainpath='./data1/0.6train.csv'
    else:
        trainpath='./data1/0.3train.csv'
else:
    testpath='./data2/test.csv'
    ori_path='./data2/unlabel_data.csv'
    if opt.label_size = 0.9
        trainpath='./data2/0.9train.csv'
    elif opt.label_size = 0.6
        trainpath='./data2/0.6train.csv'
    else:
        trainpath='./data2/0.3train.csv'


# In[ ]:


get_ipython().system('git clone https://github.com/PaddlePaddle/ERNIE.git')
get_ipython().system('git -C ./ERNIE branch origin/dygraph')
sys.path.append('./ERNIE')
from numpy import *
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import paddle as P
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModelForSequenceClassification
from ernie.modeling_ernie import ErnieModelForSequenceClassification_noisy

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn.metrics as sm

import itertools
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


# In[ ]:


#parameters of ernie
BATCH = 64
MAX_SEQLEN = 78
base_lr = 5e-5
EPOCH = 20

#uda configuration
sup_path = trainpath
aug_path= ori_path
unsup_BATCH=128
aug_len=17
sup_BATCH = BATCH
uda_coeff = 1
uda_softmax_temp = 0.85
uda_confidence_thresh = 0.45

#load_weight path
params_path = "./checkpoint/"

#save_weights path
save_path = "/home/aistudio/checkpoint/"


# In[ ]:


label_nums_str = "label"+str(label_nums)
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

def make_data(path):
    data = []
    df=pd.read_csv(path)
    for i in range(len(df)):
        text,label = df.iloc[i]['contents'],int(df.iloc[i][label_nums_str])
        text_id,_ = tokenizer.encode(str(text))
        text_id = text_id[:MAX_SEQLEN]
        text_id = np.pad(text_id,[0,MAX_SEQLEN-len(text_id)],mode='constant')
        label_id = np.array(label-1)
        data.append((text_id,label_id))
        #print(text_id)
    return data

def make_data_with_zcon(path):
    data = []
    df=pd.read_csv(path)
    for i in range(len(df)):
        text,label = df.iloc[i]['zcon'],int(df.iloc[i][label_nums_str])
        text_id,_ = tokenizer.encode(str(text))
        text_id = text_id[-MAX_SEQLEN:]
        text_id = np.pad(text_id,[MAX_SEQLEN-len(text_id),0],mode='constant')
        label_id = np.array(label-1)
        data.append((text_id,label_id))
    return data

def make_data_with_fcon(path):
    data = []
    df=pd.read_csv(path)
    for i in range(len(df)):
        text,label = df.iloc[i]['fcon'],int(df.iloc[i][label_nums_str])
        text_id,_ = tokenizer.encode(str(text))
        text_id = text_id[:MAX_SEQLEN]
        text_id = np.pad(text_id,[0,MAX_SEQLEN-len(text_id)],mode='constant')
        label_id = np.array(label-1)
        data.append((text_id,label_id))
    return data


def make_data_with_context(path):
    data = []
    df=pd.read_csv(path)
    for i in range(len(df)):
        text,context,label = df.iloc[i]['contents'],df.iloc[i]['context'],int(df.iloc[i][label_nums_str])
        text_id,_ = tokenizer.encode(str(text))
        context_id,_ = tokenizer.encode(str(context))
        if context_pos == True:
            text_id = list(text_id) + [2] + list(context_id)
            text_id = text_id[:MAX_SEQLEN]
            text_id = np.pad(text_id,[0,MAX_SEQLEN-len(text_id)],mode='constant')
        else:
            text_id = list(context_id) + [2] + list(text_id)
            text_id = text_id[-MAX_SEQLEN:]
            text_id = np.pad(text_id,[MAX_SEQLEN-len(text_id),0],mode='constant')
        label_id = np.array(label-1)
        data.append((text_id,label_id))
    return data


def make_aug_data():
    data = []
    df = pd.read_csv(aug_path)
    for i in range(len(df)):
        if context_type=="fzon":
            text,label = df.iloc[i]['fcon_aug'],int(0)
            text_id,_ = tokenizer.encode(str(text))
            text_id = text_id[:aug_len]
            text_id = np.pad(text_id,[0,aug_len-len(text_id)],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
        elif context_type=="context":
            text,context,label = df.iloc[i]['aug'],df.iloc[i]['context_aug'],int(df.iloc[i][label_nums_str])
            text_id,_ = tokenizer.encode(str(text))
            context_id,_ = tokenizer.encode(str(context))
            if context_pos == True:
                text_id = list(text_id) + [2] + list(context_id)
                text_id = text_id[:MAX_SEQLEN]
                text_id = np.pad(text_id,[0,MAX_SEQLEN-len(text_id)],mode='constant')
            else:
                text_id = list(context_id) + [2] + list(text_id)
                text_id = text_id[-MAX_SEQLEN:]
                text_id = np.pad(text_id,[MAX_SEQLEN-len(text_id),0],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
        else:
            text,label = df.iloc[i]['zcon_aug'],int(0)
            text_id,_ = tokenizer.encode(str(text))
            text_id = text_id[-aug_len:]
            text_id = np.pad(text_id,aug_len-len(text_id),0],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
    return data

def make_ori_data():
    data = []
    df = pd.read_csv(ori_path)
    for i in range(len(df)):
        if context_type=="fzon":
            text,label = df.iloc[i]['fcon'],int(df.iloc[i][label_nums_str])
            text_id,_ = tokenizer.encode(str(text))
            text_id = text_id[:aug_len]
            text_id = np.pad(text_id,[0,aug_len-len(text_id)],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
            
        elif context_type=="context":
            text,context,label = df.iloc[i]['contents'],df.iloc[i]['context'],int(df.iloc[i][label_nums_str])
            text_id,_ = tokenizer.encode(str(text))# ErnieTokenizer
            context_id,_ = tokenizer.encode(str(context))# ErnieTokenizer
            if context_pos == True:
                text_id = list(text_id) + [2] + list(context_id)
                text_id = text_id[:MAX_SEQLEN]
                text_id = np.pad(text_id,[0,MAX_SEQLEN-len(text_id)],mode='constant')
            else:
                text_id = list(context_id) + [2] + list(text_id)
                text_id = text_id[-MAX_SEQLEN:]
                text_id = np.pad(text_id,[MAX_SEQLEN-len(text_id),0],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
        else:
            text,label = df.iloc[i]['zcon'],int(0)
            text_id,_ = tokenizer.encode(str(text))
            text_id = text_id[-aug_len:]
            text_id = np.pad(text_id,aug_len-len(text_id),0],mode='constant')
            label_id = np.array(label-1)
            data.append((text_id,label_id))
    return data


# In[ ]:


if use_uda == False:
    if use_context == False:
        test_data = make_data(testpath)
        train_data = make_data(trainpath)
    else:
        if context_type == "context":
            test_data = make_data_with_context(testpath)
            train_data = make_data_with_context(trainpath)
        elif context_type == "zcon":
            test_data = make_data_with_zcon(testpath)
            train_data = make_data_with_zcon(trainpath)
        else:
            test_data = make_data_with_fcon(testpath)
            train_data = make_data_with_fcon(trainpath)
else:
    if use_context == False:
        test_data = make_data(testpath)
        sup_data = make_data(trainpath)
    else:
        if context_type == "context":
            test_data = make_data_with_context(testpath)
            sup_data = make_data_with_context(sup_path)
        elif context_type == "zcon":
            test_data = make_data_with_zcon(testpath)
            sup_data = make_data_with_zcon(sup_path)
        else:
            test_data = make_data_with_fcon(testpath)
            sup_data = make_data_with_fcon(sup_path)
    ori_data = make_ori_data()
    aug_data = make_aug_data()

#print(test_data[:2])


# In[ ]:


def get_batch_data(data, i):
    d = data[i*BATCH: (i + 1) * BATCH]
    feature, label = zip(*d)
    feature = np.stack(feature)

    label = np.stack(list(label))

    feature = D.to_variable(feature)
    label = D.to_variable(label)
    return feature, label

def get_unsup_batch_data(data, i):
    d = data[i*unsup_BATCH: (i + 1) * unsup_BATCH]
    feature, label = zip(*d)
    feature = np.stack(feature)

    feature = D.to_variable(feature)
    return feature

def get_sup_batch_data_repeat(data, i): 
    i = i % (len(sup_data) // sup_BATCH)
    d = data[i*sup_BATCH: (i + 1) * sup_BATCH]
    feature ,label = zip(*d)
    feature = np.stack(feature)
    label = np.stack(list(label))
    feature = D.to_variable(feature)
    label = D.to_variable(label)
    return feature, label

def kl_for_log_probs(log_p, log_q):
    p = L.exp(log_p)
    neg_ent = L.reduce_sum(p * log_p, dim=-1)
    neg_cross_ent = L.reduce_sum(p * log_q, dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl


# In[9]:


import six
if six.PY2:
    from pathlib2 import Path
else:
    from pathlib import Path

params_path = Path(params_path)
D.guard().__enter__()

max_f1 = 0

if use_uda == False:
    dd = len(train_data) // BATCH
else:
    dd = len(sup_data) // BATCH
LR= F.layers.cosine_decay( learning_rate = base_lr, step_each_epoch=dd, epochs=EPOCH)

if label_nums == 7:
    if load_weight == True:
        ernie = ErnieModelForSequenceClassification.from_pretrained(params_path, num_labels=label_nums)
    else:
        ernie = ErnieModelForSequenceClassification.from_pretrained('ernie-1.0', num_labels=label_nums)
else:
    if load_weight == True:
        ernie = ErnieModelForSequenceClassification_noisy.from_pretrained(params_path, num_labels=label_nums)
    else:
        ernie = ErnieModelForSequenceClassification_noisy.from_pretrained('ernie-1.0', num_labels=label_nums)

optimizer = F.optimizer.Adam(LR, parameter_list=ernie.parameters())

train_loss0=[]
test_loss0=[]

train_f10=[]
test_f10=[]
train_f1_wei=[]
test_f1_wei=[]

train_precision_wei=[]
test_precision_wei=[]
train_precision_macro=[]
test_precision_macro=[]

train_recall_wei=[]
test_recall_wei=[]
train_recall_macro=[]
test_recall_macro=[]

train_acc=[]
test_acc=[]

sup_loss0=[]
unsup_loss0=[]

for i in range(EPOCH):
    if use_uda == False:
        np.random.shuffle(train_data)
        print("EPOCH "+str(i+1)+":")
        #train
        train_pred0=[]

        for j in range(len(train_data) // BATCH):
            feature, label = get_batch_data(train_data, j)
            loss, logits1= ernie(feature, labels=label)

            if use_focal == True:
                loss = focal_loss(logits1, label, alpha, epsilon = epsilon, gamma = gamma)

            loss.backward()

            train_pred=L.argmax(logits1, -1).numpy()
            train_pred0.append(train_pred)

            train_f1=f1_score(label.numpy(), train_pred, average='macro')
            train_f1_w=f1_score(label.numpy(), train_pred, average='weighted')

            train_precision_m=precision_score(label.numpy(), train_pred, average='macro')
            train_precision_w=precision_score(label.numpy(), train_pred, average='weighted')

            train_recall_m=recall_score(label.numpy(), train_pred, average='macro')
            train_recall_w=recall_score(label.numpy(), train_pred, average='weighted')

            train_acc0=accuracy_score(label.numpy(), train_pred)

            optimizer.minimize(loss)
            ernie.clear_gradients()
            #if j % 10 == 0:
              #  print('train %d: loss %.5f' % (j, loss.numpy()))
            # evaluate

            if j % 100 == 0:
                j1=j
                train_loss0.append(loss.numpy())
                train_f10.append(train_f1)
                train_f1_wei.append(train_f1_w)
                train_precision_macro.append(train_precision_m)
                train_precision_wei.append(train_precision_w)
                train_recall_macro.append(train_recall_m)
                train_recall_wei.append(train_recall_w)
                train_acc.append(train_acc0)

                print('train %d: loss %.5f' % (j, loss.numpy()))
                print('train %d: f1 %.5f' % (j, train_f1))
                print('train %d: acc %.5f' % (j1, train_acc0))
                all_pred, all_label = [], []

                loss1=[]
                with D.base._switch_tracer_mode_guard_(is_train=False): 
                    ernie.eval() 
                    for j in range(len(test_data) // BATCH):
                        feature, label = get_batch_data(test_data, j)
                        loss, logits = ernie(feature, labels=label) 
                        all_pred.extend(L.argmax(logits, -1).numpy())
                        all_label.extend(label.numpy())

                        loss1.append(loss.numpy())
                    ernie.train()

                test_loss=np.concatenate(loss1).mean()
                test_acc0=accuracy_score(all_label, all_pred)

                test_f1 = f1_score(all_label, all_pred, average='macro')
                test_f1_w=f1_score(all_label, all_pred,average='weighted')

                test_precision_m=precision_score(all_label, all_pred,average='macro')
                test_precision_w=precision_score(all_label, all_pred,average='weighted')

                test_recall_m=recall_score(all_label, all_pred,average='macro')
                test_recall_w=recall_score(all_label, all_pred,average='weighted')

                #print('f1 %.5f' % f1)
                test_loss0.append(test_loss)
                test_f10.append(test_f1)
                test_f1_wei.append(test_f1_w)
                test_precision_macro.append(test_precision_m)
                test_precision_wei.append(test_precision_w)
                test_recall_macro.append(test_recall_m)
                test_recall_wei.append(test_recall_w)
                test_acc.append(test_acc0)
                print('test %d: loss %.5f' % (j1, test_loss))
                print('test %d: f1 %.5f' % (j1, test_f1))
                print('test %d: acc %.5f' % (j1, test_acc0))

                if save_weights == True:
                    if test_f1 > max_f1:
                        max_f1 = test_f1
                        print("Save weights.")
                        params_list = ['classifier.weight' , "classifier.bias"]
                        '''
                        opt_list = ["linear_73.w_0_AdamOptimizer_0_moment1_0",
                        "linear_73.b_0_AdamOptimizer_0_moment1_0",
                        "linear_73.w_0_AdamOptimizer_0_moment2_0",
                        "linear_73.b_0_AdamOptimizer_0_moment2_0"]
                        '''
                        params_dict = ernie.state_dict()
                        #opt_dict = optimizer.state_dict()
                        [params_dict.pop(k) for k in params_list]
                        #[opt_dict.pop(k) for k in opt_list]

                        F.save_dygraph(params_dict, save_path+'/saved_weights')
                        #F.save_dygraph(opt_dict, save_path+'/saved_weights')

                print()
    else: # UDA uda_softmax_temp uda_confidence_thresh
        np.random.shuffle(sup_data)

        print("EPOCH "+str(i+1)+":")
        #train
        train_pred0=[]
        ori_pred=[]
        aug_pred=[]

        for j in range(len(aug_data) // unsup_BATCH):
            #sup_loss
            sup_feature, sup_label = get_sup_batch_data_repeat(sup_data, j)
            sup_loss, sup_logits1= ernie(sup_feature, labels=sup_label)
            train_pred=L.argmax(sup_logits1, -1).numpy()
            train_pred0.append(train_pred)
            label = sup_label
            train_f1=f1_score(label.numpy(), train_pred, average='macro')
            train_f1_w=f1_score(label.numpy(), train_pred, average='weighted')

            train_precision_m=precision_score(label.numpy(), train_pred, average='macro')
            train_precision_w=precision_score(label.numpy(), train_pred, average='weighted')

            train_recall_m=recall_score(label.numpy(), train_pred, average='macro')
            train_recall_w=recall_score(label.numpy(), train_pred, average='weighted')

            train_acc0=accuracy_score(label.numpy(), train_pred)

            #if use_focal == True:
                #sup_focal_loss = focal_loss(sup_logits1, sup_label, alpha, epsilon = epsilon, gamma = gamma)
            
            sup_log_probs = L.log_softmax(sup_logits1, axis=-1)
            if len(sup_label.shape) == 1:
                    sup_label = L.reshape(sup_label, [-1, 1])

            one_hot_labels = L.one_hot(input=sup_label, depth=label_nums)
            tgt_label_prob = one_hot_labels
            per_example_loss = -L.reduce_sum(tgt_label_prob * sup_log_probs, dim=-1)

            loss_mask = L.ones_like(per_example_loss)
            correct_label_probs = L.reduce_sum(one_hot_labels * L.exp(sup_log_probs), dim=-1)
            loss_mask.stop_gradient=True
            per_example_loss = per_example_loss * loss_mask
            sup_loss = (L.reduce_sum(per_example_loss) /
                    max(L.reduce_sum(loss_mask), 1))
            #if use_focal == True:
            #    sup_loss = sup_loss + sup_focal_loss

            #unsup_loss
            #ori
            with D.guard():
                ori_input_ids = get_unsup_batch_data(ori_data, j)##ori_data

                ori_loss, ori_logits = ernie(ori_input_ids, labels=None)##ernieClassifier
                #ori_prob = L.softmax(ori_logits, axis=-1)    # KLdiv target
                #ori_log_prob = L.log_softmax(ori_logits, axis=-1)    # KLdiv target

                if uda_softmax_temp != -1:
                    ori_log_prob = L.log_softmax(ori_logits / uda_softmax_temp,axis=-1)
                    ori_log_prob.stop_gradient=True
                else:
                    ori_log_prob = ori_logits
                    ori_log_prob.stop_gradient=True

            #aug
            # softmax temperature controlling
            #uda_softmax_temp = uda_softmax_temp if uda_softmax_temp > 0 else 1

                aug_input_ids = get_unsup_batch_data(aug_data, j)##aug_data

                aug_loss, aug_logits = ernie(aug_input_ids, labels=None)
                aug_log_prob = L.log_softmax(aug_logits / uda_softmax_temp, axis=-1)

            ori_pred.extend(L.argmax(ori_log_prob, -1).numpy())
            aug_pred.extend(L.argmax(aug_log_prob, -1).numpy())

            if uda_confidence_thresh != -1:
                largest_prob = L.reduce_max(L.exp(ori_log_prob),dim=-1)
                uda_confidence_thresh_tensor = L.fill_constant(shape=largest_prob.shape, value=uda_confidence_thresh, dtype='float32')
                unsup_loss_mask = L.cast(L.greater_than(x = largest_prob,y = uda_confidence_thresh_tensor),dtype="float32")
                unsup_loss_mask.stop_gradient=True

            per_example_kl_loss = kl_for_log_probs(ori_log_prob, aug_log_prob) * unsup_loss_mask
            unsup_loss = L.reduce_mean(per_example_kl_loss)

            #final loss 
            loss = sup_loss + uda_coeff*unsup_loss
            loss.backward()

            optimizer.minimize(loss)
            ernie.clear_gradients()

            if j % 100 == 0:
                j1=j
                train_loss0.append(loss.numpy())
                #train_f10.append(train_f1)
                #train_acc.append(train_acc0)
                sup_loss0.append(sup_loss.numpy())
                unsup_loss0.append(unsup_loss.numpy())
                train_f10.append(train_f1)
                train_f1_wei.append(train_f1_w)
                train_precision_macro.append(train_precision_m)
                train_precision_wei.append(train_precision_w)
                train_recall_macro.append(train_recall_m)
                train_recall_wei.append(train_recall_w)
                train_acc.append(train_acc0)

                print('train %d: loss %.5f' % (j, loss.numpy()))
                print('train %d: f1 %.5f' % (j, train_f1))
                print('train %d: acc %.5f' % (j1, train_acc0))
                print('train %d: sup_loss %.5f' % (j, sup_loss.numpy()))
                print('train %d: unsup_loss %.5f' % (j, unsup_loss.numpy()))

                all_pred, all_label = [], []

                loss1=[]

                with D.base._switch_tracer_mode_guard_(is_train=False):
                    ernie.eval()
                    for j in range(len(test_data) // sup_BATCH):
                        feature, label = get_batch_data(test_data, j)
                        loss, logits = ernie(feature, labels=label) 

                        all_pred.extend(L.argmax(logits, -1).numpy())
                        all_label.extend(label.numpy())

                        loss1.append(loss.numpy())
                    ernie.train()

                test_loss=np.concatenate(loss1).mean()
                test_f1 = f1_score(all_label, all_pred, average='macro')
                test_acc0=accuracy_score(all_label, all_pred)
                test_f1_w=f1_score(all_label, all_pred, average='weighted')
                test_precision_m=precision_score(all_label, all_pred, average='macro')
                test_precision_w=precision_score(all_label, all_pred, average='weighted')
                test_recall_m=recall_score(all_label, all_pred, average='macro')
                test_recall_w=recall_score(all_label, all_pred, average='weighted')

                test_loss0.append(test_loss)
                test_f10.append(test_f1)
                test_acc.append(test_acc0)
                test_f1_wei.append(test_f1_w)
                test_precision_macro.append(test_precision_m)
                test_precision_wei.append(test_precision_w)
                test_recall_macro.append(test_recall_m)
                test_recall_wei.append(test_recall_w)

                print('test %d: loss %.5f' % (j1, test_loss))
                print('test %d: f1 %.5f' % (j1, test_f1))
                print('test %d: acc %.5f' % (j1, test_acc0))

                if save_weights == True:
                    if test_f1 > max_f1:
                        max_f1 = test_f1
                        print("Save weights.")
                        params_list = ['classifier.weight' , "classifier.bias"]
                        '''
                        opt_list = ["linear_73.w_0_AdamOptimizer_0_moment1_0",
                        "linear_73.b_0_AdamOptimizer_0_moment1_0",
                        "linear_73.w_0_AdamOptimizer_0_moment2_0",
                        "linear_73.b_0_AdamOptimizer_0_moment2_0"]
                        '''
                        params_dict = ernie.state_dict()
                        #opt_dict = optimizer.state_dict()
                        [params_dict.pop(k) for k in params_list]
                        #[opt_dict.pop(k) for k in opt_list]

                        F.save_dygraph(params_dict, save_path+'/saved_weights')
                        #F.save_dygraph(opt_dict, save_path+'/saved_weights')
                
                print()


# In[ ]:





# In[ ]:


ranges=range(len(train_loss0)) # Get number of epochs
if use_uda == False:
    plt.plot(ranges, train_loss0, 'r')
    plt.plot(ranges, train_f10, 'b')
    plt.plot(ranges, train_acc, 'y')
    plt.title('Training loss, f1 and acc')
    plt.xlabel("EPOCH")
    plt.ylabel("f1-loss-acc")
    plt.legend(["train_loss", "train_f1","train_acc"])
    plt.xticks(range(0,len(train_loss0),EPOCH),range(1,EPOCH+1))
    
    plt.figure()
else:
    plt.plot(ranges, train_loss0, 'r')
    #plt.plot(ranges, train_f10, 'b')
    #plt.plot(ranges, train_acc, 'y')
    plt.plot(ranges, sup_loss0, 'b')
    plt.plot(ranges, unsup_loss0, 'y')

    plt.title('Training loss,sup_loss,unsup_loss')
    plt.xlabel("EPOCH")
    plt.ylabel("loss-sup loss-unsup loss")
    plt.legend(["train_loss","sup_loss","unsup_loss"])
    plt.xticks(range(0,len(train_loss0),EPOCH),range(1,EPOCH+1))
    #plt.vlines(epoch_line, 0, max(train_loss0), colors = "grey", linestyles = "dashed")
    #plt.figure(figsize=(6,6))
    plt.figure()


# In[ ]:


#test
plt.plot(ranges, test_loss0, 'r')
plt.plot(ranges, test_f10, 'b')
plt.plot(ranges, test_acc, 'y')
plt.title('Test loss, f1 and acc')
plt.xlabel("EPOCH")
plt.ylabel("f1-loss-acc")
plt.legend(["test_loss", "test_f1","test_acc"])
plt.xticks(range(0,len(train_loss0),EPOCH),range(1,1+EPOCH))
#plt.grid(axis="x")
plt.figure()


# In[ ]:


#confusion_matrix：
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : Calculated value of confusion matrix
    - classes
    - normalize : True/False
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    #plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_y=all_label
pred_test_y=all_pred
m = sm.confusion_matrix(test_y, pred_test_y)
print('confusion_matrix：', m, sep='\n')
attack_types = [str(k) for k in range(1,label_nums+1)]
plot_confusion_matrix(m, classes=attack_types, normalize=True, title='Normalized confusion matrix')


# In[ ]:


# log

if use_uda == False:
    ll=[]
    for i in list(range(1,EPOCH+1)):
        ll+=[i]*len(range(0,len(train_data) // BATCH,100))
    l = list(range(0,len(train_data) // BATCH,100))*EPOCH
    train_loss0 = list(map(float, train_loss0))
    log_dict = {'epoch':ll,'range':l,'train_loss':train_loss0,'train_acc':train_acc,'train_f1_macro':train_f10,
    'train_f1_weighted':train_f1_wei,'train_precision_weighted':train_precision_wei,'train_precision_macro':train_precision_macro,
    'train_recall_weighted':train_recall_wei,'train_recall_macro':train_recall_macro,
    'test_loss':test_loss0,'test_acc':test_acc,'test_f1_macro':test_f10,'test_f1_weighted':test_f1_wei,'test_precision_weighted':test_precision_wei,
    'test_precision_macro':test_precision_macro,'test_recall_weighted':test_recall_wei,'test_recall_macro':test_recall_macro}
else:
    ll=[]
    for i in list(range(1,EPOCH+1)):
        ll+=[i]*len(range(0,len(aug_data) // unsup_BATCH,100))
    l = list(range(0,len(aug_data) // unsup_BATCH,100))*EPOCH
    train_loss0 = list(map(float, train_loss0))
    log_dict = {'epoch':ll,'range':l,'train_loss':train_loss0,'sup_loss':sup_loss0,
    'sup_loss':sup_loss0,'test_loss':test_loss0,'test_acc':test_acc,'test_f1_macro':test_f10,'test_f1_weighted':test_f1_wei,'test_precision_weighted':test_precision_wei,
    'test_precision_macro':test_precision_macro,'test_recall_weighted':test_recall_wei,'test_recall_macro':test_recall_macro}

log_dict_df = pd.DataFrame(log_dict)
log_dict_df.to_csv('/home/aistudio/log.csv',encoding = "utf-8-sig")


# In[ ]:


# pred
f=pd.read_csv(testpath)
contents=f['contents'][:len(all_label)]
test_pred = {'contents':contents,'original_label':all_label,'test_label':all_pred}

pred2 = pd.DataFrame(test_pred)
pred2.to_csv('/home/aistudio/test_pred.csv',encoding = "utf-8-sig")

