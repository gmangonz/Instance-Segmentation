# Instance-Segmentation

This repository contains my implementation of instance segmentation for the Carvana Challenge. My first attempt was done in my Introduction To Machine Learning course where I got a mean dice
score of 0.92 which means, I was withing 7% of the top performances. However, this was 2 years ago and now with new knowledge and better understanding of computer vision and Tensorflow, I would like to give it another shot. 

# Requirements
1. Python 3.7
2. Tensorflow-gpu 2.6.0
3. Tensorflow-addons
4. Keras

Additional requirements listed in requirements.yaml file.

## Installation

Recommended to use [Anaconda](https://www.anaconda.com/)

```
conda env create -f requirements.yml (TODO)
conda activate tf-gpu
```

# Usage
## Data Preparation
(in progress)

## Train
(in progress)

## Results and Submission

One of the new implementations I used for this project was adaptive data augmentation. This means that while the model trains, data augmentation will occur depending on the error produced by the model. For example, if the model is starting off and the error is high, the data augmentation will be minimal. However, once the error begins to decrease as the training continues, data augmentation will be occur at a higher frequency and different transformations will stack to add randomness. 

  Batch of images from dataset     |      Respective Masks        |
:-------------------------:|:------------------------:|
| <img src="images\data.jpg" width=500px> | <img src="images\masks.jpg" width=500px> |


  Ada with high error (low accuracy)    |      Corresponding Masks        |
:-------------------------:|:------------------------:|
| <img src="images\data_ada_high_error.jpg" width=600px> | <img src="images\mask_ada_high_error.jpg" width=600px> |

  Ada with lower error     |      Corresponding Masks        |
:-------------------------:|:------------------------:|
| <img src="images\data_ada_mid_error.jpg" width=600px> | <img src="images\mask_ada_mid_error.jpg" width=600px> |

  Ada with low error (high accuracy)      |      Corresponding Masks       |
:-------------------------:|:------------------------:|
| <img src="images\data_ada_low_error.jpg" width=600px> | <img src="images\mask_ada_low_error.jpg" width=600px> |

# Results
(In progress)