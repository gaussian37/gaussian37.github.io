---
layout: post
title: 4.1 Vector Space Model (VSM)
date: 2018-07-05 00:10:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: VSM.png # Add image post (optional)
tags: [Vector Space Model, VSM] # add tag
---
# Vector Space Model (VSM)

citation : Pro Deep Learning with Tensorflow

In NLP information-retrieval systems, a document is generally represented as simply a vector of the count of the words it contains.
For retrieving documents similar to a specific document either the <mark>cosine of the angle</mark> or the <mark>dot product</mark> between the document and other documents is computed.
The cosine of the angle between two vectors gives a similarity measure based on the similarity between their vector compositions.
To illustrate this fact, let us look at two vectors x, y $$\in$$

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$



