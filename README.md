# BSCNET
This source repository is dedicated for the following people: 
#### Jiayue Liu, Ziyao Zhou, Ming Gao, Jiafu Tang, Weiguo Fan, Aspect Sentiment Mining of Short Bullet Screen Comments from Online TV Series, 2022 
If you are interested in this research and use this code, please kindly cite our paper or contact the corresponding author.

### Highlights
#### A new dataset for aspect-level sentiment classification of BSCs with downstream tasks.
#### 7-class aspect-level senmtiment classificaiton with contextual information using a semi-supervised learning strategy.
#### Noisy BSC Identification and Popularity Prediction of TV Series by using sentiment features extracted from BSCNET.

# Usage:
## 1 To reproduce experiment 
python experiment --label_nums = 7 --dataset =  1/2 #(default = 1) --label_size =  0.3/0.6/0.9 #(default = 0.9) --context_type =  P0/P1/P2/P3/P4 #(default = P4) --use_uda = True/False #(default = True) --save_weights = True/False #(default = False) --load_weight = True/False #(default = False)
## 2 To reproduce downstream tasks
## 1 To reproduce task1
python task1 --label_nums = 2 --dataset =  1/2 #(default = 1) --label_size =  0.3/0.6/0.9 #(default = 0.9) --context_type =  P0/P1/P2/P3/P4 #(default = P4) --use_uda = True/False #(default = True) --save_weights = True/False #(default = False) --load_weight = True/False #(default = False)
## 2 To reproduce task2
python task2 --dataset = 1/2 #(default = 1) --task_type =  M0.3/M1.0/E1 #(default = M1.0)

# Datasets(You need to extract the *.zip files first):
## 1 In directory data1
There are 9 files for the train and test.
0.9train,0.6train,0.3train,text,labeled are for experiment and downstream tasks.
M0.3,M1.0,E1 are for downstream tasks.
0.3trainmeans 30% of the data are pretraind, 0.6train means 60% of the data are pretraind and 0.9train means 60% of the data are pretraind.
M0.3 are features extracted from 0.3train to predict the popularity.
E1 areaspect-level sentiment features along with descriptive features in each sliding window of 3 episodes to build the time series prediction model

## 2 In directory data2
There are 9 files for vaidation. Same as Data1

# Requirements:
Paddlepaddle=1.8.0
python==3.7
ERNIE=1.0
