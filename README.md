# BSCNET
This source repository is dedicated for the following published journal paper: 
#### Jiayue Liu, Ziyao Zhou, Ming Gao, Jiafu Tang, Weiguo Fan, Aspect Sentiment Mining of Short Bullet Screen Comments from Online TV Series, Journal of , 2022, https://doi.org/. (https://www)
If you are interested in this research and use this code, please kindly reference our paper or contact the corresponding author.

### Highlights
#### A new dataset for short text sentiment is accompanied as well as the source code.

# Usage:
## 1 To reproduce experiment 
python experiment --label_nums = 7 --dataset =  1/2 #(default = 1) --label_size =  0.3/0.6/0.9 #(default = 0.9) --context_type =  P0/P1/P2/P3/P4 #(default = P4) --use_uda = True/False #(default = True) --save_weights = True/False #(default = False) --load_weight = True/False #(default = False)
## 2 To reproduce downstreamtasks
## 1 To reproduce task1
python task1 --label_nums = 2 --dataset =  1/2 #(default = 1) --label_size =  0.3/0.6/0.9 #(default = 0.9) --context_type =  P0/P1/P2/P3/P4 #(default = P4) --use_uda = True/False #(default = True) --save_weights = True/False #(default = False) --load_weight = True/False #(default = False)
## 2 To reproduce task2
python task2 --dataset = 1/2 #(default = 1) --task_type =  M0.3/M1.0/E1 #(default = M1.0)

# Datasets(You need to extract the *.zip files first):
## 1 In directory datasets_exp1/
There are 6 different sub-datasets, whose name indicates pred_task+pred_tgt. In each sub-dataset, for example, in 3CPU/, there are 3 DAG information files(train, val, and test) and 2 performance datasets(train, val, and test) using different pred_mode(i.e., PRIOR_1 and PRIOR_ALL).

## 2 In directory datasets_exp2_3/
There are 3 different splits. In each split, there are 3 DAG information files(train, val, and test) and their corresponding performance data(train, val, and test).

# Requirements:
Paddlepaddle=1.8.0
python==3.7
ERNIE=1.0
