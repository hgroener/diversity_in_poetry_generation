# This is a fork of the repository https://github.com/b3nji87/master-thesis-diversity-in-poetry-generation
# Master Thesis: Diversity in Poetry Generation
This repository contains code that can be used to reproduce the results obtained in the master thesis project "Diversity in Poetry Generation".


## Setup
1. Clone this repository.
2. Set up a virtual environment (Python 3.10) using Conda. Activate the environment and install dependencies:
    ```
    conda create -n diversity-in-poetry-generation python=3.10
    conda activate diversity-in-poetry-generation
    pip install 'git+https://github.com/potamides/uniformers.git#egg=uniformers'
    pip install --upgrade gensim
    conda install -c conda-forge libarchive
    pip install evaluate matplotlib cydifflib lexical-diversity sentence-transformers spacy pyldavis
    ```


## Data Preparation
In order to deploy various training techniques and run experiments, training data needs to be processed in different ways. Refer to diversity-in-poetry-generation/training_data/ReadMe.md for more details and code instructions.


## Generators
After proprocessing training data, we are able to train/finetune different models in the context of poetry generation.
#### Deep-speare
Training and sampling is done through an extern [Deep-speare repository](https://github.com/b3nji87/deepspeare-fork).
#### Structured-Adversary
As for Deep-speare, we use an extern repository for training and sampling: [Structured-Adversary](https://github.com/b3nji87/Structured-Adversary-Fork).
#### Unconditioned Large Language Models (LLMs)
Refer to diversity-in-poetry-generation/training/unconditioned for coding instructions.
#### Style-conditioned LLMs
Refer to diversity-in-poetry-generation/training/style_conditioned for coding instructions.


## Sampling
diversity-in-poetry-generation/sampling/Readme.md contains code instructions to sample from different models and process the obtained samples properly.


## Analysis
We analyze different aspects of diversity in diversity-in-poetry-generation/analysis. Refer to Readme.md for detailed instructions.
