## Introduction

This project is a comprehensive research of predicting main properties of molecules from their structural characteristics. The aim of the project is to develop and test state-of-the-art graph-based machine learning models capable of predicting, with high accuracy, parameters such as toxicity and energy of molecules. The basis for molecule energy prediction was the [MEGNet](https://github.com/Gruz2520/predict_energy_of_mols) pet=project.

To achieve this goal, advanced neural network libraries and architectures specialised in working with molecular data in graph representation were used:

- The [DIG (Deep Intuition for Graphs)](https://github.com/divelab/DIG) library is a comprehensive toolkit for solving a wide range of problems related to graph structures, including molecular modelling.
- Models [ComENet](https://arxiv.org/abs/2206.08515), [DimeNet++](https://arxiv.org/abs/2011.14115), [SchNet](https://arxiv.org/abs/1706.08566), [SphereNet](https://openreview.net/forum?id=givsRXsOt9r) - advanced deep neural network architectures demonstrating high performance in tasks related to molecular structures.

A distinctive feature of this project was the development and application of novel data augmentation methods and the testing of the hypothesis that the new method would improve quality metrics.

Specifically, the targets in this project were:

- For toxicity - `mouse_intraperitoneal_LD50` (mean lethal dose to mice when administered intraperitoneally).
- For energy, `U_0` (internal energy of the molecule).

## tl;dr:
The new augmentation method did not produce increases in key metrics. However, models were trained and compared for the two tasks.
### results.
|Model|energy_MAE|toxic_MAE|
|-----|---|---|
|MEGNet|**0.0017 meV\atom**|0.5012|
|SchNet|0.0042 meV\atom|0.3985|
|DimeNet++|0.0066 meV\atom|0.3887|
|ComENet|0.0031 meV\atom|0.3980|
|SphereNet|0.0027 meV\atom|**0.3756**|

For energy, MEGNet was able to beat everyone in accuracy per atom due to its lightness. 

For toxicity, we didn't go for accuracy, although we beat [PassOnline](https://passonline.org/) 0.5 vs **0.3756** [Our SphereNet](scripts_for_train_on_cluster/sphere_100.py), which is a leader among medics for calculating compound properties. We were determining the performance of a new way of augmenting the data. So we trained each of the models and increased the number of variations of each of the molecules:
![alt text](imgs/results/output.png)
Unfortunately when increasing the number of molecules we didn't get any improvement in accuracy, so here is a graph of just one of all the models.

## Repository Architecture
The repository contains basic scripts for training, data analysis notebooks with preprocessing. Dataset, trained models, scripts for running `sbatch` to set the training task on the [HSE cluster cHARISMa](https://hpc.hse.ru/en/hardware/hpc-cluster/), etc. are not posted.

- [predict_energy.ipynb](predict_energy.ipynb) - main notebook with calculations for molecular energy datset and preparation of data for training.
- [predict_toxic.ipynb](predict_toxic.ipynb) - main notebook with calculations for the molecule toxicity dataset and data preparation for training.
- [test_model_setups_notebooks](test_model_setups_notebooks) - tested different model setsups on small samples to select hyperparameters for training before training.
- [scripts_for_train_on_cluster](scripts_for_train_on_cluster) - scripts for parallel launch of model training on the cluster.
  - [tts.py](scripts_for_train_in_on_cluster/tts.py) - was decomposing scripts to create custom train_test_split, as several partitioning variations were tested.
  - [utils.py](scripts_for_train_in_on_cluster/utils.py) - highlighted basic data preprocessing functions for ease of modification.

## Requirements
When creating the environment, it is required to build `torch` from binary files, otherwise it will not detect the GPU for training.

```bash
pip install -r req.txt
```
