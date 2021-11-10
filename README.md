## Curriculum learning for improved femur fracture classification: scheduling data with prior knowledge and uncertainty
#### by Amelia Jiménez-Sánchez, Diana Mateus, Sonja Kirchhoff, Chlodwig Kirchhoff, Peter Biberthaler, Nassir Navab, Miguel A. González Ballester, Gemma Piella
#### in Medical Image Analysis

This repository provides a TensorFlow implementation of our work -> [[**PDF**]](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003182) [[**arXiv**]](https://arxiv.org/abs/2007.16102)

## Overview 
In this paper, we propose a unified [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) formulation to schedule the order and pace of the training samples presented to the optimizer. Our novel formulation reunites three strategies consisting of individually weighting training samples, reordering the training set, or sampling subsets of data. We define two novel scoring functions: one from domain-specific prior knowledge and an original self-paced uncertainty score. We perform experiments on a clinical dataset of proximal femur radiographs. The curriculum improves proximal femur fracture classification up to the performance of experienced trauma surgeons. Using the publicly available MNIST dataset, we further discuss and demonstrate the benefits of our unified CL formulation for three controlled and challenging digit recognition scenarios: with limited amounts of data, under class-imbalance, and in the presence of label noise.

<p align="center"><img width="80%" src="abstract.svg" /></p>

In this repository, we provide the code to reproduce the experiments on digit recognition with the publicly available dataset MNIST. 

## Requirements:
- Python 3.5+
- TensorFlow 1.4+
- Sklearn
- Numpy

## Usage
### 1. Cloning the repository
```bash
$ git clone https://github.com/ameliajimenez/curriculum-learning-prior-uncertainty.git
$ cd curriculum-learning-prior-uncertainty/
```

### 2. Using prior knowledge or uncertainty
To introduce prior knowledge into the training through curriculum probabilities for sampling the training set, set `init_probs` argument with initial probabilities for each of the classes. 
```bash
$ data_provider = read_data_sets(datadir,  init_probs)
```
To dynamically estimate uncertainty, set the `compute_uncertainty` variable to "True". 


### 3. Training the model
Choose the curriculum approach by setting the `strategy` argument to "reorder", "subsets" or "weights". The `strategy` arguments controls the call to the modified `next_batch()` function of the data provider, and the loss function.

### 4. Training the model
```bash
$ python train.py
```

### 5. Evaluating the model
```bash
$ python test.py
```

## Citation
If this work is useful for your research, please cite our [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841521003182):
```
@article{JimenezSanchez2022102273,
title = {Curriculum learning for improved femur fracture classification: Scheduling data with prior knowledge and uncertainty},
journal = {Medical Image Analysis},
volume = {75},
pages = {102273},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102273},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521003182},
author = {Amelia Jiménez-Sánchez and Diana Mateus and Sonja Kirchhoff and Chlodwig Kirchhoff and Peter Biberthaler and Nassir Navab and Miguel A. {González Ballester} and Gemma Piella},
keywords = {Curriculum learning, Self-paced learning, Data scheduler, Bone fracture, X-ray, Multi-class classification, Limited data, Class-imbalance, Noisy labels},
}
```
