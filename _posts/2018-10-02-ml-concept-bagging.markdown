---
layout: post
title: Overview Bagging  
date: 2018-10-02 22:02:00
img: ml/concept/bagging/bagging.png
categories: [ml-concept] 
tags: [machine learning, ensemble, bagging] # add tag
---

All datasets are **sampling** of entire data. Let's create various classifier with various sampling dataset.
Also, with various sampling, we can make robust classifier.

![sampling](../assets/img/ml/concept/bagging/sampling.PNG)

<br>

As above, entire data can be separated to several samples. 
After ensemble of those, we can make robust classifer as third graph.
Also, we call it that several **weak classifier** makes string classifier.

![sampling2](../assets/img/ml/concept/bagging/sampling2.PNG)

<br>

Sampling reduces overfitting problem. Even though results of samples make overfitting,
Ensemble of those has good learning result. Or, sampling reduces variance.  

![bootstrap](../assets/img/ml/concept/bagging/bootstrap.PNG)

<br>

### What is the **bootstrapping**?

+ sampling with replacement in learning dataset
    - sampling n subset learning data
+ bootstrapping means sampling without adding data
+ .632 bootstrap
    - when sampling d times, each datum is sampled with 0.632 probability.
    ![632](../assets/img/ml/concept/bagging/632bootstrap.PNG)
    
<br>
    
### What is **bagging**?
+ Bagging is Bootstrap Aggregation.
+ Ensemble with n subsample
+ Assign various data to one model.
+ Proper to high variance model.(likely to get overfitting)
    - Bagging is proper ensemble for models likely to getting overfitting well.
    - Combination of each overfitting lessens this problem and works out.
+ supports regressor, classifier

![bagging](../assets/img/ml/concept/bagging/bagging2.png)

<br>

### what is **Out of bag error**
+ OOB is error estimation
+ When using Bagging, validate the performance with not included data in the bag.
    - It's similar to deal with validation set
+ Good standard for evaluating Bagging performance
    ![bagging_standard](../assets/img/ml/concept/bagging/bagging_standard.png)

<br>
    
### How to use Bagging in sklearn

![bagging_sklearn](../assets/img/ml/concept/bagging/bagging_sklearn.PNG)

+ base_estimator : One estimator
    - Concept of bagging is using one classifier and many different data
+ n_estimator : #bootstrap
+ max_samples : data ratio to use
+ max_features : feature ratio to use
    
![bagging_regressor](../assets/img/ml/concept/bagging/baggingregressor.PNG)

<br>

BaggingRegressor is same with BaggingClassifier to use.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
```

![template](../assets/img/ml/concept/bagging/template.PNG)
