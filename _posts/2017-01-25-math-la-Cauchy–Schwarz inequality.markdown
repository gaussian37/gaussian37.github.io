---
layout: post
title: 코시 슈바르츠 부등식의 증명 
date: 2017-01-25 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 벡터, 내적] # add tag
---

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

이번 글에서는 `Cauchy-Schwarz inequality`(코시 슈바르츠 부등식)에 대하여 알아보겠습니다.

<img src="../assets/img/math/la/cauchy-schwarz inequality/1.png" alt="Drawing" style="width: 600px;"/>

+ 코시 슈바르츠 부등식은 $$ \vert \vec{x} \cdot \vec{y} \vert  \le \| \vec{x} \| \ \|\vec{y}\| $$ 관계를 가집니다.
    + 좌변의 $$ \vert \vec{x} \cdot \vec{y} \vert $$ 는 `절대값`을 뜻합니다.
+ 특히, $$ \vec{x} = c\vec{y} $$ 인 경우에는 $$ \vert\vec{x} \cdot \vec{y} \vert  = \| \vec{x} \| \ \|\vec{y}\| $$ 관계를 가집니다.
+ 코시 슈바르츠 부등식의 증명을 위하여 `벡터의 길이`는 항상 `0이상의 값`을 가짐을 생각해 봅시다.
    +  벡터의 길이는 `루트`값이기 때문에 실수 범위에서는 항상 0보다 크거나 같아야 합니다.
+ 식을 증명하기 위해서 $$ p(t) = \| t\vec{y} - \vec{x} \|^{2} $$ 을 이용하겠습니다. 이 식은 `벡터의 길이` 이므로 0보다 크거나 같습니다.
+ p(t)를 전개하면 위의 슬라이드와 같이 전개할 수 있습니다.    

<br>

<img src="../assets/img/math/la/cauchy-schwarz inequality/3.png" alt="Drawing" style="width: 600px;"/>

+ 전개한 식은 p(t)에 대한 식이므로 변수 t에 대한 2차 식으로 정리하기 위하여 $$ \vec{y} \cdot \vec{y} = a , \ 2(\vec{x} \cdot \vec{y}) = b, \ \vec{x} \cdot \vec{x} = c $$ 로 치환해 보겠습니다.
+ 위의 슬라이드와 같이 전개하면 최종적으로 $$ 4ac \ge b $$로 정리할 수 있습니다.
+ 앞에서 정의한 a, b, c를 $$ 4ac \ge b $$에 대입해 보겠습니다.

<br>

<img src="../assets/img/math/la/cauchy-schwarz inequality/4.png" alt="Drawing" style="width: 600px;"/>

+ 최종적으로 식을 정리하면 $$ \vert\vec{x} \cdot \vec{y} \vert  \le \| \vec{x} \| \ \|\vec{y}\| $$ 관계를 유도할 수 있습니다.
+ 더 나아가서 $$ \vec{x} = c\vec{y} $$ 라고 정의하고 대입한 후 식을 정리하면 $$ \vert\vec{x} \cdot \vec{y} \vert  = \| \vec{x} \| \ \|\vec{y}\| $$ 관계를 유도할 수 있습니다.
+ **코시 슈바르츠 부등식은 선형대수학의 증명**에서 자주 사용됩니다.
+ 다음 강의에서는 **벡터의 내적과 코시슈바르츠 부등식**이 어떻게 사용되는지 알아보겠습니다.

<br><br>
<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>