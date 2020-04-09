---
layout: post
title: 감마분포와 지수분포 (Gamma distribution and Exponential distribution)
date: 2020-04-06 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [gamma distribution, exponential distribution, 감마분포, 지수분포 ] # add tag
---

<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

## **목차**

<br>

- ### 감마 함수
- ### 감마 분포와 지수 분포
- ### 감마 분포의 평균과 분산
- ### 지수 분포의 평균과 분산
- ### 포아송 과정과의 관계
- ### 확률 분포 정리

<br>

- 이번 글에서는 감마 분포와 지수 분포에 대하여 다루어 보려고 합니다.
- 감마 분포는 감마 함수와 연관되어 있어 이름이 감마 분포이고 지수 분포는 물론 지수 함수와 연관되어 있어 지수 분포 이름을 가지게 됩니다. 특히 지수 분포는 감마 분포의 특수한 경우에 해당됩니다.
- 감마 분포는 대표적으로 대기 시간이 얼마나 되는지, 어떤 사건이 발생할 때 까지 얼마나 많은 시간이 필요한 지 등에 사용되어 신뢰도에도 적용할 수 있습니다.

<br>

## **감마 함수**

<br>

- `감마 함수`는 $$ \alpha > 0 $$ 인 $$ \alpha $$에 대하여 다음과 같이 정의 됩니다.

<br>

$$ \Gamma(\alpha) = \int_{0}^{\infty}x^{\alpha - 1}e^{-x} dx $$

<br>

- 위 감마 함수는 대표적으로 다음 성질이 성립합니다.
- ① $$ \alpha > 1 $$일 때, $$ \Gamma(\alpha) = (\alpha -1)\Gamma(\alpha - 1) $$
- ② $$ \Gamma(1) = 1 $$이고 양의 정수  n에 대하여 $$ \Gamma(n) = (n - 1)! $$
- ③ $$ \Gamma(\frac{1}{2}) = \sqrt{\pi} $$

<br>

- 두 성질을 보면 감마 함수는 팩토리얼을 `실수` 범위로 확장한 것으로 해석할 수 있습니다.
- 그러면 위 2가지 성질에 대하여 증명을 해보도록 하겠습니다. 

<br>

- 먼저 ① 식을 증명하기 위하여 부분 적분의 성질을 이용해 보도록 하겠습니다. 다음과 같습니다.

<br>

$$ \int f'(x)g(x) = f(x)g(x) - \int f(x)g'(x) $$

<br>

- 여기서 적분 하기 쉬운 것을 $$ f(x) $$로 두겠습니다. $$ f(x) = e^{-x} $$, $$ g(x) = x^{\alpha -1} $$

<br>

$$ \Gamma(\alpha) = -x^{\alpha - 1}e^{-x}\vert^{\infty}_{0} + \int^{\infty}_{0} (\alpha - 1)x^{\alpha - 2}e^{-x} dx $$

$$ = (\alpha - 1)\Gamma(\alpha -1) $$

<br>

- 다음으로 ② 식을 증명해 보도록 하겠습니다.

<br>

$$ \Gamma(1) = \int_{0}^{\infty}e^{-x} dx = -e^{-x} \vert_{0}^{\infty} = 1 $$

<br>

$$ \Gamma(n) = (n - 1)\Gamma(n-1) = (n - 1)(n - 2)\Gamma(n-3) = (n - 1)(n - 2) ... (1)\Gamma(1) = (n - 1)! $$

<br>

- 마지막으로 ③ 식에 대하여 다루어 보겠습니다. 실제로 감마 함수를 사용할 때, $$ \Gamma(\frac{1}{2}) = \sqrt{\pi} $$를 많이 이용하는데 이 값이 어떻게 도출되는 지 다루어 보겠습니다. 이 값은 `치환 적분`과 `극좌표`를 이용하여 구할 수 있습니다.

<br>

$$ \Gamma(\alpha) = \int_{0}^{\infty} t^{\alpha - 1} e^{-t} dt $$

$$ \Gamma(\frac{1}{2}) = \int_{0}^{\infty} t^{-\frac{1}{2}} e^{-t}dt $$

<br>

- 이 때, $$ x = \sqrt{t} $$로 치환하면, $$ dx = \frac{1}{2}t^{-\frac{1}{2}} dt $$가 됩니다.

<br>

$$ \Gamma(\frac{1}{2}) = 2 \int_{0}^{\infty} e^{-x^{2}} dx $$

$$ \Gamma(\frac{1}{2})^{2} = 4 \int_{0}^{\infty} e^{-x^{2}} dx \int_{0}^{\infty} e^{-y^{2}} dy $$

$$ = 4\int_{0}^{\infty} \Bigl(\int_{0}^{\infty} e^{-x^{2}} \Bigr)dx \ e^{-y^{2}} dy $$

$$ = 4\int_{0}^{\infty}\int_{0}^{\infty} e^{-x^{2}}e^{-y^{2}}dx \ dy $$

$$ = 4\int_{0}^{\infty}\int_{0}^{\infty} e^{-(x^{2} + y^{2})} dx \ dy $$

<br>

- 위 식에서 사용된 $$ x, y $$를 극좌표가 사용되는 공간으로 가져와서 변환해 보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/pb/gamma_and_exponential_distribution/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 $$ x, y $$는 $$ r, \theta $$ 로 변환될 수 있고 범위도 변하게 되어 다음 식과 같아 집니다.

<br>

$$ \int\int\ f(x, y)\ dx\ dy = \int\int\ f(r\cos(\theta), r\sin(\theta))\ r\ dr \ d\theta $$

<br>

- 따라서 나머지 식을 전개하면 다음과 같이 정리됩니다.

<br>

$$ \Gamma(\frac{1}{2})^{2} = 4 \int_{0}^{\frac{\pi}{2}} \int_{0}{\infty} e^{-r^{2}}r \ dr \ d\theta $$

<br>

- 여기서 $$ \int_{0}{\infty} e^{-r^{2}}r \ dr $$ 부분과 $$ \theta $$는 관련이 전혀 없으므로 $$ \theta $$ 관련 적분 밖으로 나올 수 있습니다.

<br>

$$ = 4 \int_{0}{\infty} \Biggl( \int_{0}^{\frac{\pi}{2}}1 \ d\theta \Biggr) e^{-r^{2}}r \ dr  $$

$$ = 4 \cdot \frac{\pi}{2} \int_{0}^{\infty} re^{-r^{2}}\ dr\ d\theta $$

<br>

- 치환 적분을 통해 식을 전개 합니다. $$ u = -r^{2}, du = -2rdr $$로 치환합니다. 그러면 다음 식과 같이 됩니다.

<br>

$$ = 4 \cdot \frac{\pi}{2} \int_{0}^{\infty} re^{-u}\ \frac{-1}{2r}du $$

$$ = 4 \cdot \frac{\pi}{2} \cdot (-\frac{1}{2}) \int_{0}^{\infty} e^{-u} du $$

$$ = 4 \cdot \frac{\pi}{2} \cdot (-\frac{1}{2}) [ e^{-u} ]_{0}^{\infty} $$

$$ = \pi $$

<br>

- 따라서 양변에 루트를 적용하면 최종적으로 다음과 같이 정리 됩니다.

<br>

$$ \Gamma(\frac{1}{2}) = \sqrt{\pi} $$

<br>







<br>

## **감마 분포와 지수 분포**

<br>

<br>

## **감마 분포의 평균과 분산**

<br>

<br>

## **지수 분포의 평균과 분산**

<br>


<br>

## **포아송 과정과의 관계**

<br>

<br>

## **확률 분포 정리**

<br>


<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

