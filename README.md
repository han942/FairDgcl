# FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning

This is the PyTorch implementation for **FairDgcl** proposed in the paper **FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning**, which is submitted to the Transactions on Knowledge and Data Engineering (TKDE).


## 1. Introduction

As trustworthy AI continues to advance, the fairness issue in rec- ommendations has received increasing attention. A recommender system is considered unfair when it produces unequal outcomes for different user groups based on user-sensitive attributes (e.g., age, gender). 
Some researchers have proposed data augmentation-based methods aiming at alleviating user-level unfairness by altering the skewed distribution of training data among various user groups. De- spite yielding promising results, they often rely on fairness-related assumptions 
that may not align with reality, potentially reducing the data quality and negatively affecting model effectiveness. To tackle this issue, in this paper, we study how to implement high- quality data augmentation to improve recommendation fairness. Specifically, we propose FairDgcl, 
a dynamic graph adversarial contrastive learning framework aiming at improving fairness in recommender system. First, FairDgcl develops an adversarial con- trastive network with a view generator and a view discriminator to learn generating fair augmentation strategies in an adversarial style. 
Then, we propose two dynamic, learnable models to generate contrastive views within contrastive learning framework, which automatically fine-tune the augmentation strategies. Meanwhile, we theoretically show that FairDgcl can simultaneously generate enhanced 
representations that possess both fairness and accuracy. Lastly, extensive experimental results on four real-world datasets demonstrate the effectiveness of the proposed FairDgcl.

## 2. Running environment

We develop our codes in the following environment:

- python==3.9.13
- numpy==1.23.1
- torch==1.11.0
- scipy==1.9.1
- torch-sparse==0.6.17

## 3. How to run the codes

Due to size limitations, we have not uploaded the data. You can refer to the link in the paper to download the data and rerun the experiments.
The command lines to train FairDgcl on the three datasets are as below. 
```python
python Main.py 
```
