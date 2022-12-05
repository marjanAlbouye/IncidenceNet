# Incidence Networks for Geometric Deep Learning
This is a pytorch implementation of "Incidence Networks for Geometric Deep Learning" paper. 
Link to paper: https://arxiv.org/pdf/1905.11460.pdf

# Requirements:
 - `numpy==1.17.4`
 - `torch==1.4.0`
 - `chainer_chemistry`
 - `rdkit`
 - (optional) `torch_geometric==1.4.2` (https://github.com/rusty1s/pytorch_geometric)

# Dataset prepration

There are two ways to get the processed dataset before start running:

1- Download the dataset from the following links and save it under data direcotry.

 https://figshare.com/articles/dataset/qm9_complete_inhomo_zip/12649757 (dense graph)
 https://figshare.com/articles/dataset/Processed_QM9_sparse_/12649790 (sparse graph)
 
2- Run `qm9_prep_main.py` from data_prep directory. (It takes a bit time to process data)

# Training and Evaluation

 To train and evaluate the model on a target, run the following command at src directory.
 
 ``` python main.py --mode mode  --target_index target_index --data_path data_path --log_path log_path --is_linear is_linear  --is_sym is_sym  --graph_type graph_type```
 
 **Arguments:**
 
 ```--mode```: 0 for node-node adjacency and 1 for node-edge
 
 ```--target_index```: index to the molecular target [0-11] (see table below)
 
 ```--data_path```: path to input data
 
 ```--log_path```: path to the log (default: `../results/{inhomo/homo}_checkpoints/t{target_index}_{taget_name}/`)
 
 ```--is_linear```: 0 for non-linear and 1 for linear
 
 ```--is_sym```: 0 for symmetric adjacency and 1 for non-symmetric adjacency
 
 ```--graph_type```: dense or sparse


**Molecular target properties** 

Target index | Target name
------------ | -------------
0 | mu
1 | alpha
2 | homo
3 | lumo
4 | gap
5 | R2
6 | ZPVE
7 | U0
8 | U
9 | H
10 | G
11 | Cv
