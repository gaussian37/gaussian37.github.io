---
layout: post
title: What is a probability distribution
date: 2019-08-04 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [베이지안, bayes, 확률] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/jbhi96p4mwI" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

- 이번 글에서는 `probability distribution`에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/pb/bayes_stat_05/1.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이번 강의는 간단하게 `확률 분포`가 무엇인지 알아보는 내용이었습니다.
- 특히 이번 강의의 키워드인 `discrete`한 변수에서는 `PMF(Probability Mass Function)`이라는 용어가 나왔고 `continuous`한 변수에서는 `PDF(Probability Density Function)`라는 용어가 나왔습니다.
- 위 슬라이드에서 볼 수 있듯이 **discrete**한 변수에는 각 변수 당 발생할 확률이 정해져 있고 모든 변수가 나올 확률을 다 더했을 때 1이 되어야 합니다.
    - 즉, $$ \sum_{x} pmf(x) = 1 $$ 이 됩니다.
- 위 예제에서는 모든 변수가 발생할 확률이 동등하고 이런 경우 `uniform`하다고 합니다.

<br>

- 반면, **continuous**한 변수에서는 각 변수 당 발생할 확률은 0에 수렴합니다. 위 슬라이드의 예 처럼 키가 정확히 1.73m 인 사람을 뽑을 확률은 거의 0에 가깝습니다. 1.731m, 1.732m인 사람은 해당되지 않으니까요.
- 이렇게 **continuous**한 변수를 가지는 분포에서는 정확히 어떤 변수에 대하여 확률이 대응되기 보다는 `구간`을 이용하여 확률을 구합니다.
- 따라서 **PDF**의 **D**가 뜻하는 **Density**처럼 얼마나 빽빽하게 분포되어 있는 지에 해당하는 밀도가 바로 확률에 해당합니다.
- 물론 **PDF**에서도 전체 **Density**를 합하면 1이 되어야 합니다.
    - 즉, $$ \int_{-infty}^{infty} PDF(x) dx = 1 $$이 되어야 합니다.