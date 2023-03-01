# BSCNET
This source repository is dedicated to the following article : 
#### Jiayue Liu, Ziyao Zhou, Ming Gao*, Jiafu Tang, Weiguo Fan, Aspect Sentiment Mining of Short Bullet Screen Comments from Online TV Series, 2022 
If you are interested in this research and wish to use these datasets and codes, please kindly cite our paper or contact the corresponding author(*).

### Highlights
#### A new dataset for aspect-level sentiment classification of  Bullet Screen Comments(BSCs) from TV series “Put your head on my shoulder”(致我们暖暖的小时光) and "the Bad Kids" (隐秘的角落), along with downstream tasks.
#### A novel aspect-level sentiment analysis framework of BSCs (BSCNET) combining pre-trained Transformer-based encoder, text context and semi-supervised learning.
#### Noisy BSC identification and popularity prediction of future episodes by using sentiment features extracted from BSCNET.

# Pictures
## A TV series’ screenshot from the Tencent video platform.
![A TV series’ screenshot from the Tencent video platform.](https://github.com/cloudworkflow/BSCNET/blob/main/img/p3.png)

## The proposed aspect-level sentiment analysis framework.
![The proposed aspect-level sentiment analysis framework.](https://github.com/cloudworkflow/BSCNET/blob/main/img/p4.png)
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
There are 9 files for training and testing.

The files `0.9train.csv`,`0.6train.csv`,`0.3train.csv`,`test.csv`,`labeled.csv` are used for experiment and downstream tasks.

The files `M0.3.csv`,`M1.0.csv`,`E1.csv` are for downstream tasks.

`0.3train` means 30% of the data are pretraind, `0.6train` means 60% of the data are pretraind and `0.9train` means 60% of the data are pretraind.

`M0.3` represents the features extracted from `0.3train` to predict the popularity.

`E1.csv` contains aspect-level sentiment features along with descriptive features in each sliding window of 3 episodes to build the time series prediction model.


## 2 In directory data2
There are 9 files for vaidation. Same as Data1.

# Requirements:
Paddlepaddle=1.8.0
python==3.7
ERNIE=1.0
