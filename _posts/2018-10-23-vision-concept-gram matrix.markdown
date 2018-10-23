---
layout: post
title: Gram Matrix used in Style Transfer  
date: 2018-10-23 05:20:00
img: vision/concept/gram-matrix/grammatrix.png
categories: [vision-concept] 
tags: [gram matrix, style transfer] # add tag
---

Today, we are going to study about `gram matrix` used in [Style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

What is the `gram matrix`?

Let $$ \vec{x} $$ be a flatten image vector. (even though in this example it has only 3 elements for simplicity.)

![img1](../assets/img/vision/concept/gram-matrix/img1.png)

<br>

The shape of image is not important because as preprocessomg we will flatten matrix/tensor to vector.
Accordingly, principle of applying `gram matrix` is same with following method.

Let $$ Z_{0}, Z_{1} $$ be filters which applying to vector $$ \vec{x} $$.
In order to make `gram matrix`, we will follow below procedure.

1. Apply $$ \vec{x} $$ to $$ Z_{0}, Z_{1} $$. In below example, N = #filters = 2 & M = #pixel = 3. 

![img2](../assets/img/vision/concept/gram-matrix/img2.png)

<br>

2. Calculate `gram matrix`

![img3](../assets/img/vision/concept/gram-matrix/img3.png)

![img4](../assets/img/vision/concept/gram-matrix/img4.png)

<br>

'gram matrix' means the relation between filters.
It looks similar with `correlation`.

![img5](../assets/img/vision/concept/gram-matrix/img5.png)

The difference between `gram matrix` and `correlation` is whether to subtract `mean` before multiplying.
(In `gram matrix` there is no subtraction.)
But, like a `correlation`, `gram matrix` also means the relation between two distributions(filters).

Okay! This concept is very important to understand the `Neural Style Transfer`.

Thanks for reading.