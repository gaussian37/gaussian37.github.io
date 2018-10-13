---
layout: post
title: Information Theory (Entropy, KL Divergence)  
date: 2018-10-13 08:42:00
img: ml/concept/Information-Theory/Cross-Entropy_print.png
categories: [ml-concept] 
tags: [Information Theory, Entropy, KL divergence] # add tag
---

Let's think about two cases. Which case does have **more information**?

1. It's clear today. In the news, it will be clear tomorrow too.
2. It's clear today. In the news, it will be heavy rain tomorrow.

In the first case, probability of clear in tomorrow is high because today is clear.
Intuitively, we get less information. On the contrary, probability of second case is low.
But we get information a lot. By this simple example, we can figure out that probability and information gain has inverse proportional relation.
Information gain can be though as **alarming rate**.
For example, If world war 3 happens, we will be alarmed too much. (we gain a lot of data..)

Basically, an incident which happens in low probability makes us alarmed. And it has a lot of data.
This relation is the main concept of `information theory`.

The main function of `information theory` is quantifying the information and making possible to calculate it.
What we can measure is `probability`.

we can formulate above example like this. 

$$ P(tomorrow = heavy_rain | today = clear) , P(tomorrow = clear | today  = clear) $$

Okay! we can define `how to measure information quantity` when knowing the probability of an incident.

$$ h(x) = -log_{2}P(x) \tag{1} $$




 

