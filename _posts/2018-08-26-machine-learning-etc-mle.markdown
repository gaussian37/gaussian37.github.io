---
layout: post
title: What is the Maximum Likelihood Estimation 
date: 2018-08-26 23:13:00
img: machine-learning/etc/mle/mle.jpg
categories: [ml-concept] 
tags: [machine learning, mle, maximum likelihood estimation] # add tag
---

+ likelihood : A probability of happening possibility of an event.

Let's start with Bernoulli distribution !!

+ Bernoulli Distribution :  

![1](../assets/img/machine-learning/etc/mle/1.PNG)

![2](../assets/img/machine-learning/etc/mle/2.png)

In this notation, We calculate the probability given x and theta. <br>
Object is finding `θ` which maximize the probability.

![3](../assets/img/machine-learning/etc/mle/3.png)

In the bernoulli distribution, let me suppose that p is sigmoid.

![4](../assets/img/machine-learning/etc/mle/4.PNG)

And then, we can get likelihood with taking the log function.

Why do we take a log function? Because earlier notation, To get the probability, we multiplied all probabilities. <br>
Multiplying too many probabilities makes the result too much small. Therefore, we transform the multiply to add by log function. <br>

![5](../assets/img/machine-learning/etc/mle/5.PNG), 

![6](../assets/img/machine-learning/etc/mle/6.png)

![7](../assets/img/machine-learning/etc/mle/7.PNG)

Wow, It's same with `cost function of logistic regression`.

![8](../assets/img/machine-learning/etc/mle/8.PNG)

Surely, partial derivation of MLE is same with partial derivation of cost function.

### Remember! <br>

+ MLE and Cost function have same result in logistic regression !!
    - Therefore, we can use cost function as maximize the parameter θ.