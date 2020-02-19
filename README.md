# Incidence Networks for Geometric Deep Learning
This is a pytorch implementation of the Incidence Networks for Geometric Deep Learning paper. (Submitted to ICML 2020)

# Requirements:
 - `numpy==1.17.4`
 - `torch==1.4.0`
 - `torch_geometric==1.4.2` (https://github.com/rusty1s/pytorch_geometric)

# Dataset prepration

There are two ways to get the processed dataset before start running:

1- Download the dataset from https://figshare.com/s/dad7db4fa42cb6f0fdd2 and save it under data direcotry. Note that this link only contains one version of processed dataset. If you want to train all different configurations of the models we tried, you need to run the script to prepare datasets. (details are given in option 2)

2- Run the following command at data_prep directory.

```python qm9_prep.py --graph_type graph_type --adj_type adj_type```

**Arguments:**

```--graph_type```: "sparse" or "dense"

```--adj_type```: for node-node adjency set to "h" and for node-edge set "i"

This script first downloads the raw QM9 dataset from (https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip) and stores the processed dataset that contains all the adjacecny and node/edge features under data directory.

# Training and Evaluation

 To train and evaluate the model on a target run the following command at src directory.
 
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
