## Curriculum learning for improved femur fracture classification: scheduling data with prior knowledge and uncertainty
#### by Amelia Jiménez-Sánchez, Diana Mateus, Sonja Kirchhoff, Chlodwig Kirchhoff, Peter Biberthaler, Nassir Navab, Miguel A. González Ballester, Gemma Piella

This repository provides a TensorFlow implementation of our work -> [[**arXiv**]](https://arxiv.org/abs/2007.16102)

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
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/2007.16102):
```
@article{jimenez2020curriculum,
  title={Curriculum learning for annotation-efficient medical image analysis: scheduling data with prior knowledge and uncertainty},
  author={Jim{\'e}nez-S{\'a}nchez, Amelia and Mateus, Diana and Kirchhoff, Sonja and Kirchhoff, Chlodwig and Biberthaler, Peter and Navab, Nassir and Ballester, Miguel A Gonz{\'a}lez and Piella, Gemma},
  journal={arXiv preprint arXiv:2007.16102},
  year={2020}
}
```
