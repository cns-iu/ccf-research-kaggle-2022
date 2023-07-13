# Segmenting functional tissue units across human organs using community-driven development of generalizable machine learning algorithms

The development of a reference atlas of the healthy human body requires automated image segmentation of major anatomical structures across multiple organs based on spatial bioimages generated from various sources with differences in sample preparation. We present the setup and results of the “Hacking the Human Body” machine learning algorithm development competition hosted by the Human Biomolecular Atlas (HuBMAP) and the Human Protein Atlas (HPA) teams on the Kaggle platform. We showcase how 1,175 teams from 78 countries engaged in community-driven, open-science code development that resulted in machine learning models which successfully segment anatomical structures across five organs using histology images from two consortia and that will be productized in the HuBMAP data portal to process large datasets at scale in support of Human Reference Atlas construction. We discuss the benchmark data created for the competition, major challenges faced by the participants, and the winning models and strategies.

The repo is structured in the following way:
```
├── models
├── utils
```
## Data

All data used in the competition and the external data used by the winning teams can be [accessed here](https://doi.org/10.5281/zenodo.7545745). 

## Models

This repository contains the code for the three performance prize winners.

All trained models (baseline model, winning algorithms) can be [accessed here](https://doi.org/10.5281/zenodo.7545793).

All instructions on running the three winning models can be found in the `models` directory.

A version of this code is archived on Zenodo at the time of publication at [https://doi.org/10.5281/zenodo.8144891](https://doi.org/10.5281/zenodo.8144891). Additionally, it contains the source data files for the figures/plots presented in the paper, and the scores and metrics of the top 50 teams in the competition which are used for the statistical analysis.
