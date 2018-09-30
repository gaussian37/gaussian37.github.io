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

You’ll sometimes hear that deep learning only works when lots of data is available.
This is valid in part: one fundamental characteristic of deep learning is that it can find interesting features in the training data on its own, without any need for manual feature engineering,
and this can only be achieved when lots of training examples are available.
This is especially true for problems where the input samples are very high-dimensional, like images.

But what constitutes lots of samples is relative—relative to the size and depth of the network you’re trying to train, for starters.
It isn’t possible to train a convnet to solve a complex problem with just a few tens of samples, 
but a few hundred can potentially suffice if the model is small and well regularized and the task is simple.
Because convnets learn local, translation-invariant features, they’re highly data efficient on perceptual problems. 
Training a convnet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You’ll see this in action in this section.

What’s more, deep-learning models are by nature highly repurposable: 
you can take, say, an image-classification or speech-to-text model trained on a large-scale dataset and reuse it on a significantly different problem with only minor changes.
Specifically, in the case of computer vision, many pretrained models (usually trained on the Image-Net dataset) are now publicly available for download and can be used to bootstrap powerful vision models out of very little data.
That’s what you’ll do in the next section. Let’s start by getting your hands on the data.

## 5.2.2. Downloading the data

The Dogs vs. Cats dataset that you’ll use isn’t packaged with Keras. It was made available by Kaggle as part of a computer-vision competition in late 2013, back when convnets weren’t mainstream.
You can download the original dataset from [www.kaggle.com/c/dogs-vs-cats/data](www.kaggle.com/c/dogs-vs-cats/data)

The pictures are medium-resolution color JPEGs.

+ Samples from the Dogs vs. Cats dataset. Sizes weren’t modified: the samples are heterogeneous in size, appearance, and so on.

![5.8](../assets/img/dl/chollet/05-2/05fig08_alt.jpg)

Unsurprisingly, the dogs-versus-cats Kaggle competition in 2013 was won by entrants who used convnets.
The best entries achieved up to 95% accuracy. 
In this example, you’ll get fairly close to this accuracy (in the next section), even though you’ll train your models on less than 10% of the data that was available to the competitors.

This dataset contains 25,000 images of dogs and cats (12,500 from each class) and is 543 MB (compressed). 
After downloading and uncompressing it, you’ll create a new dataset containing three subsets: a training set with 1,000 samples of each class, a validation set with 500 samples of each class, and a test set with 500 samples of each class.

After uncompressing data, Directory structure is below.(Before run below code, cats_and_dogs_small is empty)
```python

├───cats_and_dogs_small
│   ├───test
│   │   ├───cats
│   │   └───dogs
│   ├───train
│   │   ├───cats
│   │   └───dogs
│   └───validation
│       ├───cats
│       └───dogs
├───test1
└───train

```

```python

import os, shutil

# Path to the directory where the original dataset was uncompressed
original_dataset_dir = './train'

# Directory where you’ll store your smaller dataset
base_dir = './cats_and_dogs_small'
os.mkdir(base_dir)

# Directories for the training, validation, and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copies the first 1,000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copies the first 1,000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 500 dog images to validation_dogs_dir 
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

```
<br>

As a sanity check, let’s count how many pictures are in each training split (train/validation/test):

```python

>>> print('total training cat images:', len(os.listdir(train_cats_dir)))
total training cat images: 1000
>>> print('total training dog images:', len(os.listdir(train_dogs_dir)))
total training dog images: 1000
>>> print('total validation cat images:', len(os.listdir(validation_cats_dir)))
total validation cat images: 500
>>> print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
total validation dog images: 500
>>> print('total test cat images:', len(os.listdir(test_cats_dir)))
total test cat images: 500
>>> print('total test dog images:', len(os.listdir(test_dogs_dir)))
total test dog images: 500

```

So you do indeed have 2,000 training images, 1,000 validation images, and 1,000 test images. 
Each split contains the same number of samples from each class: this is a balanced binary-classification problem, which means classification accuracy will be an appropriate measure of success.


 





 

  
 