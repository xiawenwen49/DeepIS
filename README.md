# DeepIS: Susceptibility Estimation on Social Networks (WSDM 2021)
<!--#### -->

#### Authors: Wenwen Xia, Yuchen Li, Jun Wu, Shenghong Li
#### Please contact xiawenwen@sjtu.edu.cn for any questions.

## Introduction
Influence diffusion estimation is a crucial problem in social network analysis.
Most prior works mainly focus on predicting the total influence spread, i.e., the expected number of influenced nodes given an initial set of active nodes (aka. seeds).

While in this work, we propose the DeepIS model, which leverages graph neural networks (GNNs) for predicting susceptibility, i.e., the probability of being influenced for each node, given seeds.

DeepIS mainly constitute two components/steps when training or making predictions.
(1) a coarse-grained step where we estimate each node's susceptibility using features and NNs;
(2) a fine-grained step where we aggregate neighbors' coarse-grained susceptibility estimations to compute the fine-grained estimate for each node.  
The two modules are trained in an end-to-end manner. 


**We illustrate the train/evaluate details in the Jupyter Notebook sample_deepis.ipynb.**

### DeepIS architecture

![architecture](architecture.png?raw=true "Network architecture")

### Requirements

* python >= 3.7

* Dependency

```{bash}
scipy==1.5.0
torch==1.6.0
ipdb==0.13.4
numpy==1.18.5
scikit_learn==0.23.2
```

## Cite us

```
@inproceedings{deepis_wsdm20,
title={DeepIS: Susceptibility Estimation on Social Networks},
author={Wenwen Xia and Yuchen Li and Jun Wu and Shenghong Li},
booktitle={Proceedings of the Fourteenth ACM International Conference on Web Search and Data Mining (WSDM '21)}
year={2021}
}
```


