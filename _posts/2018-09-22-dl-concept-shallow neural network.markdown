---
layout: post
title: shallow neural network  
date: 2018-09-22 03:49:00
img: dl/concept/shallow-neural-network/deeplearningai.jpg
categories: [dl-concept] 
tags: [deep learning, andrew ng] # add tag
---

### Explanation for vectorized implementation

In this lecture, we study vectorized implementation. It's simple. with vetorization, we don't need to use for-loop for all data.
we just define Data `X`, Weight `W`, (+ bias) and do the math with `Numpy`.
To understand the below, recap the notation.

$$ a^[1](i) $$ : activation result of `1st` layer and `i-th` element(node). 

![Explanation for vectorized implementation1](../assets/img/dl/concept/shallow-neural-network/Explanation for vectorized implementation1.PNG)

![Explanation for vectorized implementation2](../assets/img/dl/concept/shallow-neural-network/Explanation for vectorized implementation2.PNG)

