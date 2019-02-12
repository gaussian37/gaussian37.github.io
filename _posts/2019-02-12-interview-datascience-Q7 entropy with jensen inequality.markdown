---
layout: post
title: Q7) 얀센 부등식을 이용한 증명 문제
date: 2018-12-30 00:00:00
img: interview/datascience/likelihood/ds_question.jpg
categories: [interview-datascience] 
tags: [interview, datascience, sample covariance, n-1] # add tag
---

두 확률 벡터 $$ p = (p_{1}, p_{2}, ..., p_{n}) $$ 이고 $$ r = (r_{1}, r_{2}, ..., r_{n}) $$ 가 있고 D(p, r) 이 다음과 같을 때,

$$ D(p,r) = \sum_{j=1}^{n} p_{j}log_{2}\frac{1}{r_{j}} - \sum_{j=1}^{n} p_{j}log_{2}\frac{1}{p_{j}} $$

D(p,r)이 음수가 아님을 Jensen's inequality를 이용하여 증명하여라.

---

Jensen\'s inequality를 이용하기 위하여 아래로 볼록 함수의 형태인 $$ -log_{2}x $$ 함수를 이용하여 부등식을 만들어 증명합니다.

$$ D(p,r) = -sum_{j=1}^{n} p_{j} log_{2} \frac{r_{j}}{p_{j}} \ge -log_{2}\sum_{j=1}^{n}p_{j} \frac{r_{j}}{p_{j}} = -log_{2}\sum_{j=1}^{n} r_{j} = -log_{2} 1 = 0 $$

$$ \therefore D(p,r) \ge 0 $$

