# NSCI0017 DRX Dissertation – Supplementary Data and Code

This repository contains the supplementary data and code supporting the MSc Research Project dissertation:

**Data-driven Screening and Experimental Validation of Phase Formation in Li-ion Disordered Rocksalt Ni–Mn–Ti Cathodes**

The repository includes:
- the curated DRX literature dataset;
- training, validation and test data splits;
- filtered CBFV composition feature matrices;
- feature-filtering outputs;
- machine-learning model comparison results;
- Extra Trees validation and test results;
- feature-importance outputs;
- predictions for the 12 experimental Ni–Mn–Ti candidates;
- Python notebooks and scripts used for data preprocessing, model training and prediction;
- software requirements for reproducing the computational workflow.

The computational workflow used a fixed random seed of 42 where supported. StandardScaler was fitted using the training set only and subsequently applied to the validation set, test set and experimental candidates.

This repository is provided as supplementary material for the dissertation.
