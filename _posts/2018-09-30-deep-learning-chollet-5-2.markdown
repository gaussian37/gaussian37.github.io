---
layout: post
title: 5-2. Training a convnet from scratch on a small dataset
date: 2018-09-30 09:28:00
img: dl/chollet/chollet.png
categories: [dl-chollet] 
tags: [deep learning, chollet, convnet] # add tag
---

Having to train an image-classification model using very little data is a common situation, which you’ll likely encounter in practice if you ever do computer vision in a professional context.
A “few” samples can mean anywhere from a few hundred to a few tens of thousands of images.
As a practical example, we’ll focus on classifying images as dogs or cats, in a dataset containing 4,000 pictures of cats and dogs (2,000 cats, 2,000 dogs).
We’ll use 2,000 pictures for training—1,000 for validation, and 1,000 for testing.

In this section, we’ll review one basic strategy to tackle this problem: <br>
training a new model from scratch using what little data you have.
You’ll start by naively training a small convnet on the 2,000 training samples, without any regularization, to set a baseline for what can be achieved.
This will get you to a classification accuracy of 71%.
At that point, the main issue will be overfitting. 
Then we’ll introduce **data augmentation**, a powerful technique for mitigating overfitting in computer vision.
By using data augmentation, you’ll improve the network to reach an accuracy of 82%.

In the next section, we’ll review two more essential techniques for applying deep learning to small datasets: <br>
**feature extraction with a pretrained network** (which will get you to an accuracy of 90% to 96%) and **fine-tuning a pretrained network** (this will get you to a final accuracy of 97%).
Together, these three strategies— <br>

+ training a small model from scratch
+ doing feature extraction using a pretrained model
+ fine-tuning a pretrained model

will constitute your future toolbox for tackling the problem of performing image classification with small datasets.

## 5.2.1. The relevance of deep learning for small-data problems

  
 