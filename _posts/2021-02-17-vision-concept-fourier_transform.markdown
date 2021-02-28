---
layout: post
title: Fourier transform (푸리에 변환)
date: 2021-02-17 00:00:00
img: vision/concept/fourier_transform/0.png
categories: [vision-concept] 
tags: [fourier transform, 퓨리에 변환] # add tag
---

<br>

- 참조 : Introduction to Computer Vision
- 참조 : https://youtu.be/TB1A2-Db67s
- 참조 : https://www.youtube.com/watch?v=spUNpyF58BY
- 참조 : https://www.youtube.com/playlist?list=PL5yujGYFVt0DhDXdKkFeou8zSV-KjOlAG (공돌이의 수학정리노트)

<br>

- 이번 글에서는 기본적인 퓨리에 변환 (Fourier transform)에 대하여 다루어 보도록 하겠습니다.
- 보다 자세한 내용의 퓨리에 변환은 아래 링크를 참조해 주시기 바랍니다. 이 글은 신호와 시스템 전반적인 내용을 다루며 그 중 퓨리에 변환에 대한 자세한 내용을 확인하실 수 있습니다.
    - 링크 : [https://gaussian37.github.io/vision/signal/](https://gaussian37.github.io/vision/signal/)

<br>

## **목차**

<br>

- ### **푸리에 급수를 배우는 이유**
- ### **푸리에 급수의 의미와 주파수 분석에서의 활용**
- ### **연속 시간 푸리에 급수 유도**
- ### **이산 시간 푸리에 급수 유도**
- ### **연속 시간 푸리에 변환 유도**
- ### **이산 시간 푸리에 변환 유도**
- ### **MATLAB 및 Python에서의 Fast Fourier Transform**
- ### **푸리에 변환에서 음의 주파수**
- ### **라플라스 변환과 푸리에 변환**

<br>

## **푸리에 급수를 배우는 이유**


## **푸리에 급수의 의미와 주파수 분석에서의 활용**

<br>

- 아래 두가지 식은 푸리에 급수에 관련된 동일한 식을 다른 관점으로 표현한 식입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 식의 관점은 **연속 신호(함수)는 무한 차원 벡터이고, 이것은 기저 벡터의 선형 결합으로 재구성할 수 있음**입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위 식의 관점은 **신호를 구성하는 각 기저 벡터는 얼마 만큼의 기여도를 가지고 있는지를 뜻**하며 이는 `주파수 분석에 활용 가능`합니다.

<br>

- 푸리에 급수에 대하여 알아보기 이전에 `벡터`와 `직교`에 대하여 간략하게 알아보도록 하겠습니다.

<br>

- 벡터 $$ \overrightarrow{i}, \overrightarrow{j} $$가 직교한다면 내적 $$ \overrightarrow{i} \cdot \overrightarrow{j} = 0 $$을 만족합니다.
- 두 벡터가 직교한다는 의미는 두 벡터의 내적이 0임을 뜻합니다. `벡터의 내적`은 두 벡터가 얼마나 닮았는 지를 뜻하기 때문에 두 벡터의 내적이 0이되어 직교한다는 뜻은 두 백터가 전혀 닮지 않았다는 뜻이되고 바꿔 말하면 두 벡터가 서로 독립적이라는 말이 됩니다.

<br>

- $$ \vec{a} \cdot \vec{b} = \vert \vec{a} \vert \vert \vec{b} \vert \cos{\theta} $$

<br>

- 이렇게 직교하는 벡터들을 이용하여 단위 벡터를 만들 수 있는데, 2차원에서 2개의 단위 벡터가 있으면 2차원 공간의 모든 벡터를 표현할 수 있습니다. 같은 개념으로 **N차원 공간의 모든 벡터를 표현하기 위해서는 N개의 직교하는 벡터가 필요**합니다.

<br>

- 앞으로 다룰 내용은 함수이기 때문에 벡터가 아닌 함수를 다루어야 합니다. 함수에서도 벡터의 직교와 같은 개념이 있을까요?
- 먼저 함수는 일반화된 벡터로 취급할 수 있습니다. N차원 벡터는 N개의 숫자 나열이듯이 실수 함수는 실수 값을 무한 개 나열한 것으로 본다면 **실수 함수는 무한 차원 벡터**라고 볼 수 있습니다. 따라서 앞으로는 **실수 (혹은 복소수) 함수는 무한차원 벡터**로 생각하도록 하곘습니다.
- 이러한 무한 차원의 벡터 공간을 표현하기 위해서 `함수의 직교`를 이용해야 합니다. 직교하는 함수를 무한 개 찾을 수 있다면 이 함수를 이용해서 어떤 실수 (혹은 복소수) 함수라도 그 구간 내의 함수를 표현할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 함수의 내적은 위 식과 같이 나타낼 수 있습니다. 이는 벡터의 내적과 개념이 유사합니다.
- 벡터의 내적을 할 때 두 벡터 $$ (a, b), (c, d) $$를 내적하면 $$ ac + bd $$가 됩니다. 이와 같이 함수의 내적을 할 때에도 두 함수를 특정 범위 내에서 함수값을 곱해주게 됩니다.
- 위 식을 보면 함수 $$ f_{2}(x) $$에는 * 즉, 켤레복소수를 적용해 주도록 되어있습니다. 이는 함수값이 복소수(복소 벡터)일 때, 계산이 실수 공간에서 만족해야 하는 내적의 공리와 연관되어 있기 때문입니다. 따라서 계산을 실수화 하기 위함이라고 보시면 됩니다.

<br>

- $$ z = a + jb $$

- $$ \vert z \vert^{2} = a^{2} + b^{2} = (a + jb)(a - jb) = a^{2} + b^{2} = z \cdot \bar{z} $$

<br>

- 복소수의 크기(절대값)을 구할 때에도 켤레 복소수를 통하여 실수화 한 것과 동일한 방식입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 만약 두 함수가 직교한다는 것은 두 함수의 내적이 0임을 뜻합니다. 이는 벡터의 직교와 같은 의미입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 무한 차원의 공간을 표현하기 위해서는 무한 차원의 직교하는 함수들이 필요하다고 앞에서 언급하였습니다.
- 범위를 줄여서 (a, b) 구간을 표현하기 위해서는 이 범위에서 직교하는 함수들이 필요합니다. 이 때 사용 될 직교 함수들의 집합을 Orthogonal Set이라고 하며 그 표기법은 위 식과 같습니다. 위 식과 같이 직교하는 함수들의 집합을 통하여 (a, b) 구간 표현할 수 있습니다.
- 따라서 같은 구간에서 정의된 함수 $$ f(x) $$는 이 직교 함수들의 대수 합으로 나타낼 수 있습니다.

<br>

- $$ f(x) = c_{0}\phi_{0}(x) + c_{1}\phi_{1}(x) + c_{2}\phi_{2}(x) + \cdots + c_{n}\phi_{n}(x) + \cdots $$

- $$ f(x) = \sum_{n=0}^{\infty}c_{n}\phi_{n}(x) $$

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 살펴본 푸리에 급수의 식을 함수 직교와 연관하여 다시 살펴보도록 하겠습니다. 먼저 위 식에서 지수 함수부에 해당하는 식을 자세히 살펴보면

<br>

- $$ \text{exp}(j \frac{2\pi k}{T} t) $$

<br>

- 이산 시간 푸리에 급수에서 위 지수부에 해당하는 식이 orthogonal set이 됩니다.
- 이 값은 오일러 공식 $$ \text{exp}(j\theta) = \cos{\theta} + j\sin{\theta} $$을 이용하면 다음과 같이 변환되어 사용되어 집니다.
    - 오일러 공식 참조 : [https://gaussian37.github.io/math-calculus-euler_formula/](https://gaussian37.github.io/math-calculus-euler_formula/)

<br>

- $$ \text{exp}(j \frac{2\pi k}{T} t) = \cos{\frac{2\pi k}{T} t} + j\sin{\frac{2\pi k}{T} t} $$

- $$ x(t) = \sum_{k=-\infty}^{\infty}a_{k}(\cos{\frac{2\pi k}{T} t} + j\sin{\frac{2\pi k}{T} t}) $$

<br>

- 즉, 어떤 임의의 신호(함수값)은 `cos`, `sin` 함수의 합으로 표현할 수 있음을 알 수 있습니다.
- 이 식에서 가장 중요한 것은 지수부의 식 $$ \text{exp}(j \frac{2\pi k}{T} t) $$가 Orthogonal set인 지 확인하는 것입니다. 따라서 집합 $$ \{\phi_{k}(t) \vert \phi_{k}(t) = \text{exp}(j \frac{2\pi k}{T}t), \ \ k = \text{integer on } [0, T] \} $$이 orthogonal set인 지 확인해 보겠습니다.

<br>

- 위 집합의 임의의 두 원소인 함수 $$ \phi_{k}(t) = \text{exp}(j \frac{2\pi \color{red}{k}}{T} t) $$와 $$ \phi_{p}}(t) = \text{exp}(j \frac{2\pi \color{blueL{p}}{T} t) $$가 있을 때, 두 함수의 내적을 해보겠습니다.

<br>

- $$ \begin{align} \int_{0}^{T} \phi_{k}(t) \phi_{p}^{*}(t) dt &= \int_{0}^{T} \text{exp}(j \frac{2\pi k}{T} t) \text{exp}(-j \frac{2\pi k}{T} t) \\ &= \int_{0}^{T} \text{exp}(j \frac{2\pi (k-p}{T} t) = \Biggl[ \frac{T}{j 2\pi (k-p)} \text{exp}(j \frac{2\pi (k-p}{T} t) \Biggr]_{0}^{T} \\ &= \frac{T}{j 2\pi (k-p)} [ \text{exp}(j 2\pi(k-p)) - 1] = 0 \quad (\because \cos{2n\pi} = 0 \text{ and } \sin{2n\pi} = 1) \end{align} $$

<br>

- 따라서 서로 다른 $$ k $$와 $$ p $$에 대하여 두 함수가 내적이기 때문에 위 집합은 orthogonal set임을 확인할 수 있습니다.
- 처음 위 식을 설명할 때 다음과 같은 문구와 같이 설명하였었습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 식의 관점은 **연속 신호(함수)는 무한 차원 벡터이고, 이것은 기저 벡터의 선형 결합으로 재구성할 수 있음**입니다.
- 이 때, `기저 벡터`는 `삼각 함수`이므로 $$ [0, T] $$ 구간 사이에 정의된 어떤 주기 신호는 삼각 함수의 선형 결합으로 재구성 할 수 있음을 확인하였습니다.

<br>

- 다음으로 아래 식과 뜻에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위 식의 관점은 **신호를 구성하는 각 기저 벡터는 얼마 만큼의 기여도를 가지고 있는지를 뜻**하며 이는 `주파수 분석에 활용 가능`합니다.

<br>

- 위 $$ a_{k} $$ 식에 대하여 생각해 보기 위해 주기 $$ T = 1 $$인 임의의 신호 $$ x(t) $$에 대해서, $$ k=1 $$일 때, $$ a_{k} $$의 의미를 살펴보겠습니다.

<br>

- $$ a_{1} = \int_{0}^{1} x(t) \text{exp}(-j 2\pi t) dt = \int_{0}^{1} x(t) \times \{\cos{2\pi t - j \sin{2\pi t}}  \} dt $$

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 식에서 `cos` 함수는 1Hz를 가지고 `sin` 함수는 3Hz를 가진다고 가정을 하여 식을 정하면 위 그래프와 같이 나타낼 수 있습니다.
- 왼쪽의 큰 그래프는 함수 $$ x(t) $$를 뜻하며 수식에 해당하는 그래프를 그리면 위 그래프와 같습니다.
- 이 $$ x(t) $$ 함수와 $$ \cos{2 * \pi * 1 * t}, \sin{2 * \pi * 1 * t} $$를 각각 `내적` 후 `적분`을 취하면 각각 0.5, 0이 됨을 계산할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 상세 계산 과정 : [링크](https://drive.google.com/file/d/1Tln1XM0vjqQiyXfx1KdoM4vCrvDqUo_G/view?usp=sharing)

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/9.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 상세 계산 과정 : [링크](https://drive.google.com/file/d/1jId3wh6cVkkbs5hmM3fmE_bD2tE6M8VW/view?usp=sharing)

<br>

- 앞에서 설명하였듯이 `내적`은 **두 벡터 또는 함수가 얼마나 닮았는 지** 나타냅니다. 위 식에 이 개념을 대입하면 **$$ x(t) $$의 삼각함수 성분 중 kernel의 성분을 얼만큼 내포**하고 있는 지 뜻하게 됩니다. 여기서 kernel은 $$ x(t) $$에 곱해지는 $$ \cos{2 \pi t}, \sin{2 \pi t} $$에 해당합니다.

<br>

- 위 계산은 $$ a_{1} $$의 경우에 해당합니다. 계산해야 하는 전체 영역은 음의 무한대에서 양의 무한대이기 때문에 전 과정을 계산을 해야 푸리에 급수를 나타낼 수 있습니다. 따라서 몇 가지 $$ a_{k} $$를 좀 더 다루어 보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프는 $$ a_{2} $$를 나타냅니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프는 $$ a_{0} $$을 나타냅니다. cos, sin 함수가 상수로 수렴되기 때문에 계산이 쉽게 가능합니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번에는 음의 방향으로 계산을 해보겠습니다. 위 그래프는 $$ a_{-1} $$을 나타냅니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이와 같은 계산을 -3 ~ 3까지 적용하였을 때, 위 테이블과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 테이블의 $$ a_{k} $$의 크기를 확인하기 위하여 절대값을 취한 뒤 그래프로 나타나면 위 그래프와 같습니다.
- 여기서 주파수를 구해보겠습니다. 일반적으로 $$ \cos{2 \pi \color{red}{f} t} $$에서 $$ \color{red}{f} $$가 주파수를 뜻합니다.

<br>

- $$ \text{exp}(-j \frac{2\pi k}{T} t) = \cos{\frac{2\pi \color{red}{k}}{\color{red}{T}} t } -j\sin{\frac{2\pi \color{red}{k}}{\color{red}{T}}t } $$

<br>

- 위 식과 같이 `cos`, `sin` 함수를 정의 할 수 있기 때문에 주파수는 $$ k / T $$가 되고 $$ T = 1$$로 가정을 하였기 때문에 주파수는 $$ k / T = k $$가 되어 가로축이 `주파수 Hz`가 됨을 알 수 있습니다. 따라서 그래프를 조금 수정하면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 그래프와 같이 $$ a_{k} $$를 이용하여 가로축이 주파수인 `주파수 분석`을 할 수 있음을 의미합니다.
- 위 그래프를 해석하면 1, -1 Hz 성분은 0.5씩 존재하고 3, -3 Hz 성분은 0.4씩 존재한다는 것을 확인할 수 있습니다. 여기서 **음의 주파수의 의미는 이 글에 뒷부분에서 다루겠습니다.**
- 이 값을 처음에 다룬 $$ x(t) = \cos{2 \pi \color{red}{t}} + 0.8 \sin{2 \pi \color{blue}{3t}} $$와 연계해서 보면 `cos`의 `1Hz` 성분 $$ \color{red}{t} $$를 음의 주파수, 양의 주파수 반반씩 0.5로 나타내어 지고 (0.5 * 2 =  1), `sin`의 `3Hz` 성분 $$ \color{blue}{t} $$를 음의 주파수, 양의 주파수 반반씩 0.4로 나타내어 집니다. (0.4 * 2 = 0.8)
- 여기서 나타나는 $$ 

<br>

## **연속 시간 푸리에 급수 유도**

<br>

<br>

## **이산 시간 푸리에 급수 유도**

<br>