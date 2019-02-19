---
layout: post
title: 베타 분포
date: 2019-02-06 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [통계학, 수리통계학, 베타 분포, 감마 함수, beta distribution] # add tag
---

+ 출처 : 
    + Probability & Statistics for Engineers & Scientists. 9th Edition.(Walpole 저. PEARSON) 
    + 수리통계학 (김수택 저. 자유 아카데미)

+ [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-probability-table/)


<br>

### 베타 함수의 정의

+ 베타 함수는 $$ \alpha, \beta > 0 $$ 일 때, $$ B(\alpha, \beta) = \int_{0}^{1} x^{\alpha-1}(1-x)^{\beta-1} dx $$ 로 정의 됩니다.

<br>

### 베타 함수의 성질

+ ·$$ B(\alpha, \beta) = B(\beta, \alpha) $$
    + 베타 함수에서 $$ \alpha, \beta $$는 서로 자리를 바꾸어도 치환을 하여 적분식을 풀면 결과는 같습니다.
+ ·$$ B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)} $$
    + ·$$ \Gamma(\alpha) = \int_{0}^{\infty}e^{-s}s^{\alpha-1}ds $$ 정의를 따르면,
    + ·$$ \Gamma(\alpha)\Gamma(\beta) = \int_{0}^{\infty}e^{-s}s^{\alpha-1}ds \int_{0}^{\infty}e^{-t}t^{\beta-1}dt$$
    + ·$$ = \int_{0}^{\infty}\int_{0}^{\infty} e^{-(s+t)}s^{\alpha-1}t^{\beta-1} dsdt $$
    + 변수변환으로 $$ s = xy, t = x(1-y) $$ 라고 하면 치환 적분처럼 생각할 수 있습니다.
        + s, t에 대한 x, y의 야코비안을 구해보겠습니다.
     
  
