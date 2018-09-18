---
layout: post
title: Overview feature engineering  
date: 2018-09-15 03:49:00
img: machine-learning/concept/feature-engineering/feature.png
categories: [ml-concept] 
tags: [machine learning, feature, feature engineering] # add tag
---

Let's study about overview of **feature engineering.**
I will talk about the shadow color subject below.

## Feature Engineering
+ Generation
    - Binarization, Quantization
    - Scaling (normalization)
    - `Interaction features`
    - `Log transformation`
    - `Dimension reduction`
    - `Clustering`
    - ...
     
+ Selection
    - `Univariate feature selection`
    - `Model-based selection`
    - `Iterative feature selection`
    - Feature removal
    - ...
    
### Interaction features

- create new features with combination of existing features.
- need to pre-knowledge and understanding of existing features.
    - ex) height, width → square (square = height * width)
    - ex) sensor1 + sensor2 → new sensor feature
    
### Log transformations

+ Data distribution is merged extremely in a point.(ex. Poisson)
    - login count, sales rate, searched word, the number of friends
    - [Log transformation practice](https://nbviewer.jupyter.org/github/gaussian37/Machine-Learning/blob/master/Feature%20Engineering/Log%20transformation%20for%20performance%20tuning.ipynb)

      ![longtail](../assets/img/machine-learning/concept/feature-engineering/longtail.PNG)    
        
+ Normal distribution is suitable for linear model.
    - If data have Poisson distribution, make them to fit the linear model. Then, data will fit to normal distribution.
     (Poission → Normal dist)       
    
      ![linear-gaussian](../assets/img/machine-learning/concept/feature-engineering/linear-gaussian.PNG)
      
### Dimension reduction
+ is used when existing feature space is much large.
+ algorithm reduces the space.
+ ex) **PCA**, **t-SNE**, LDA, ...  
    - [PCA practice](https://nbviewer.jupyter.org/github/gaussian37/Machine-Learning/blob/master/Feature%20Engineering/PCA%20with%20Iris%20data%20set.ipynb)
    
    
### Clustering
+ without Y, create the component of dataset
    - created components are used as features for classification
+ is useful when you know the relationship between data to some degree.
+ ex) **K-means** ...

### Feature selection
+ All features are not necessary to learn a model.
+ Some features are malignant to learn.
+ Too many features causes overfitting.
+ According to learning model, select necessary features and remove unnecessary features.

### Univariate feature selection
+ selects optimal feature based on statistical model.
    -  ex) Chi square, F-test, ANOVA ...
    - [univariate feature selection](https://nbviewer.jupyter.org/github/gaussian37/Machine-Learning/blob/master/Feature%20Engineering/Univariate%20select%20example.ipynb)
+ analyzes the relation between y and one feature.
+ is useful for **linear model.**
+ is fast-applicable and simple technique.
    
### Model based feature selection
+ A model find proper features in the learning process
    - ex) L1 penalty, Tree-based model
    - [Model based feature selection](https://nbviewer.jupyter.org/github/gaussian37/Machine-Learning/blob/master/Feature%20Engineering/model%20based%20feature%20select.ipynb)
+ Possible to select the features based on `Feature importance`
+ Useful to use Model-based feature selection as preprocessor of feature selection for other model
+ Tree-based ensemble also has characteristics similar to Model-based feature selection
+ Mainly used for `explanation of which features are important`.