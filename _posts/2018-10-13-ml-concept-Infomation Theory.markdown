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
2. It's clear today. In the news, it will be heavy rainy tomorrow.

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

$$ P(tomorrow = heavy_rainy | today = clear) , P(tomorrow = clear | today  = clear) $$

Okay! we can define `how to measure information quantity` when knowing the probability of an incident.

$$ h(x) = -log_{2}P(x) \tag{1} $$

equation (1) shows the answer. x is random variable. 
In above example, x is random variable showing weather of clear or rainy.
Let me suppose that x has a specific value. P(x) is x's probability and h(x) is the information quantity or, **self-information**.

For example, x has an incident `e` and its probability is P(e) = 1/1024 (only a time happens during 1024 times),
information quantity is $$ -log_{2} (1/1024) = 10 bit $$ . Extreme case is P(e) = 1. In this extreme case, we can only get the information that `e` always happens.
So, we don't get any alarming information if we additionally get to know that `e` happens. If we assign the P(e) = 1 into equation (1), 
information quantity $$ h(x) = 0 $$.

we can change base 2 to e(natural constant) then, unit is changed from `bit` to `natural`.

### What is the entropy ?

`Entropy` is the average of information quantities that random variable x can have.

$$ H(x) = -\Sum_{x}P(x)log_{2}P(x) \tag{2} $$

$$ H(x) = -\int_{\infty}^{\infty} P(x)log_{2}P(x) dx \tag{3} $$

Equation (2) is the entropy of dicrete case and (3) is of continuous case.