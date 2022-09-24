# BSCNET
This source repository is dedicated for the following published journal paper: 
#### Jiayue Liu, Ziyao Zhou, Ming Gao, Jiafu Tang, Weiguo Fan, Aspect Sentiment Mining of Short Bullet Screen Comments from Online TV Series, Journal of , 2022, https://doi.org/. (https://www)
If you are interested in this research and use this code, please kindly reference our paper or contact the corresponding author.

### Highlights
#### A new dataset for short text sentiment is accompanied as well as the source code.

# Usage:
## 1 To reproduce experiment 1
python run_exp1 --pred_task=3/5/7 #(default=7) --pred_tgt=CPU/MEM #(default=CPU) --pred_mode=PRIOR_1/PRIOR_ALL #(default=PRIOR_ALL) --use_DAG=T/F #(default=T)
## 2 To reproduce experiment 2
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2)
## 3 To reproduce experiment 3
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer/GCN --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2) --GCN_mode=bidirect/unidirect #(default=bidirect)

# Datasets(You need to extract the *.zip files first):
## 1 In directory datasets_exp1/
There are 6 different sub-datasets, whose name indicates pred_task+pred_tgt. In each sub-dataset, for example, in 3CPU/, there are 3 DAG information files(train, val, and test) and 2 performance datasets(train, val, and test) using different pred_mode(i.e., PRIOR_1 and PRIOR_ALL).

## 2 In directory datasets_exp2_3/
There are 3 different splits. In each split, there are 3 DAG information files(train, val, and test) and their corresponding performance data(train, val, and test).

# Requirements:
Paddlepaddle=1.8.0
python==3.7
ERNIE=1.0
