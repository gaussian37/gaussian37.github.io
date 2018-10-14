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

Basically, an event which happens in low probability makes us alarmed. And it has a lot of data.
This relation is the main concept of `information theory`.

The main function of `information theory` is quantifying the information and making possible to calculate it.
What we can measure is `probability`.

we can formulate above example like this. 

$$ P(tomorrow = heavy_rainy | today = clear) , P(tomorrow = clear | today  = clear) $$

Okay! we can define `how to measure information quantity` when knowing the probability of an event.

$$ h(x) = -log_{2}P(x) \tag{1} $$

equation (1) shows the answer. x is random variable. 
In above example, x is random variable showing weather of clear or rainy.
Let me suppose that x has a specific value. P(x) is x's probability and h(x) is the information quantity or, **self-information**.

For example, x has an event `e` and its probability is P(e) = 1/1024 (only a time happens during 1024 times),
information quantity is $$ -log_{2} (1/1024) = 10 bit $$ . Extreme case is P(e) = 1. In this extreme case, we can only get the information that `e` always happens.
So, we don't get any alarming information if we additionally get to know that `e` happens. If we assign the P(e) = 1 into equation (1), 
information quantity $$ h(x) = 0 $$.

we can change base 2 to e(natural constant) then, unit is changed from `bit` to `natural`.

### What is the entropy ?

`Entropy` is the average of information quantities that random variable x can have.

$$ H(x) = -\sum_{x}P(x)log_{2}P(x) \tag{2} $$

$$ H(x) = -\int_{\infty}^{\infty} P(x)log_{2}P(x) dx \tag{3} $$

Equation (2) is the entropy of dicrete case and (3) is of continuous case.

![dices](../assets/img/ml/concept/Information-Theory/dices.jpg)

Let me explain `entropy` with dice. let random variable `x` as spot on a die. 
`x` can have value from 1 to 6(1,2,3,4,5,6) and each has same probability as $$ \frac{1}{6}$$.
Accordingly, `entopy` is 2.585 bits.

$$ H(x) = -\sum\frac{1}{6}log_{2}\frac{1}{6} = 2.585 bits $$

Thinking about the characteristic of `entropy`, entropy is maximized when all events which have same probability of occurrence.
In other words, the extent of chaos become maximized or `uncertainty` become maximized. 
In the dice example, we don't know which spot will we get and predicting it is very hard.
On the other hand, the case of minimum `entropy` is a event has 1(100%) probability and others have 0(0%) probability.
In this case, `entropy` is 0. `uncertainty` is nothing. Accordingly, No chaos.

How about this example? This die has special probability of $$ P(x=1) = \frac{1}{2} $$ 
and other each event has $$ P(x = 2 or 3 or ... 6) = \frac{1}{10} $$.
Is `uncertainty` increasing or decreasing comparing to normal die?

Before check it, suppose that there are two cases. each case has two probability distribution with random variable `x`.
 
![similar](../assets/img/ml/concept/Information-Theory/similar.PNG)

As above example, two probability distributions are largely overlapped. 
In other words, there are nothing special difference between two. 

![different](../assets/img/ml/concept/Information-Theory/different.PNG)

In this case, two probability distributions are little overlapped.
Accordingly, They are different.

Then, How can we know the difference as numeric value?
In order to quantify the difference, we do it with `KL divergence`(Kullback-Leibler divergence).

### What is the KL divergence?

Let's look into `KL divergence` with the above example. we can define `KL divergence` formula like this.

$$ KL(P_{1}(x), P_{2}(x)) = \sum_{x}P_{1}(x)log_{2}\frac{P_{1}(x)}{P_{2}(x)} \tag{4} $$

$$ KL(P_{1}(x), P_{2}(x)) = \int_{R_{d}}P_{1}(x)log_{2}\frac{P_{1}(x)}{P_{2}(x)}dx \tag{5} $$






















 
























 



 

