---
layout: post
title: 가우시안 관련
date: 2019-11-17 00:00:00
img: math/pb/about_gaussian/0.png
categories: [math-pb] 
tags: [gaussian, 가우시안, 가우스 적분, 가우스 분포 공식] # add tag
---

<br>

- 공학에서 가우시안은 너무나 중요한 개념입니다. 공부하면서 종종 쓰였단 가우시안의 개념들을 정리해 놓는 글입니다.

<br>

## **목차**

<br>

- ### 가우스 함수
- ### 가우스 적분 증명
- ### 가우시안 분포 공식 증명
- ### covariance와 zero-mean gaussian의 covariance

<br><br>

## **가우스 함수**

<br>

- 가우스 함수 식은 다음과 같습니다.

<br>

$$ f(x) = a \cdot exp(-\frac{(x - b)^{2}}{c^{2}}) $$

<br>

- 여기서 $$ a (> 0), b, c $$는 실수입니다.
- 이 함수는 좌우 대칭의 **종(bell) 모양**의 곡선을 가지고 +/- 극한값을 입력으로 받으면 급격히 함수 값이 감소하게 됩니다.
- 매개변수 $$ a $$의 역할은 종 모양 곡선의 꼭대기 높이가 되고 $$ b $$는 꼭대기 중심의 위치가 됩니다. $$ c $$는 종 모양의 너비를 결정합니다.
- 가우스 함수의 의미는 **가우스 오차 함수**의 미분값(도함수)이고 가우시안 분포의 밀도 함수가 됩니다.

<br><br>

## **가우스 적분 증명**

<br>

- 위키피디아에 따른 `가우스 적분(Gaussian integral)`의 정의는 가우스 함수에 대한 실수 전체 범위의 적분으로 식은 다음과 같습니다.

<br>

$$ \int_{-\infty}^{\infty} e^{-x^{2}} dx = \sqrt{\pi} $$ 

<br>

- 가우스 적분의 증명을 이용하면 가우시안 분포 공식의 유도에도 사용할 수 있기 때문에, 가우스 적분의 증명을 어떻게 하는 지 알아보도록 하겠습니다.

<br>

$$ I = \int_{-\infty}^{\infty} e^{-x^{2}} dx $$

$$ I^{2} = \Biggl( \int_{-\infty}^{\infty} e^{-x^{2}} dx \Biggr)^{2} = \int_{-\infty}^{\infty} e^{-x^{2}} dx \int_{-\infty}^{\infty} e^{-y^{2}} dy $$

<br>

- 위 식에서 $$ x $$ 변수는 소위 말하는 더미 변수이므로 한 개의 $$ x $$를  $$ y $$ 로 변경하였습니다.

<br>

$$ \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^{2} + y^{2})} dx dy $$

<br>

- 식을 정리하면 위와 같이 두 변수 $$ x, y $$에 대하여 정리할 수 있습니다.
- 이 식을 풀기 위해 `극 좌표계` 개념을 가져오도록 하겠습니다. 극 좌표계는 $$ (r, \theta) $$로 좌표 평면의 좌표를 표현하는 방법입니다. 여기서 $$ \theta $$의 단위는 `radian`입니다. 아래 그림을 참조하시기 바랍니다.

<br>
<center><img src="../assets/img/math/pb/about_gaussian/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그러면 $$ x, y $$ 축을 이용하여 $$ (x, y) $$로 나타내는 직교 좌표계와 $$ (r, \theta) $$로 나타내는 극 좌표계의 관계를 살펴보면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/pb/about_gaussian/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 식에 따라서 $$ x^{2} + y^{2} = r^{2} $$이 성립합니다. 물론 원의 방정식에 의한 관계라고 이해하셔도 무방합니다.

<br>

$$ x^{2} + y^{2} = (r \cos{(\theta)})^{2} + (r \sin{(\theta)})^{2} = r^{2}(\cos^{2}{(\theta)} + \sin^{2}{(\theta)}) = r^{2} $$

<br>

- 이렇게 직교 좌표계를 극 좌표계로 바꾸는 이유는 **적분을 하기 위함**입니다. 즉, 적분할 때, $$ dx, dy $$를 $$ dr, d\theta $$로 바꾸려고 합니다.
- 그러면 적분의 구간은 $$ x : (-\infty, \infty), \  y : (-\infty, \infty) \to r : (0, \infty), \ \theta : (0, 2\pi)  $$로 변경됩니다.

<br>
<center><img src="../assets/img/math/pb/about_gaussian/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, 위 그림과 같이 기존의 직교 좌표계에서는 $$ x, y $$의 값이 음의 무한대에서 양의 무한대의 영역의 범위를 가지게 되므로 좌표계 전체 영역을 적분 할 수 있는 반면 극 좌표계에서는 원점에서 시작하는 선 $$ r $$의 길이가 0에서 양의 무한대의 영역의 범위를 가지고 그 선의 이동 영역이 0에서 $$ 2\pi $$ 만큼의 범위를 가지게 되므로 **똑같이 좌표계 전체 영역을 적분**할 수 있게 됩니다.
- 따라서 **적분하는 영역은 같으나 접근 방식이 다르다**고 이해하시면 됩니다.

<br>

- 그러면 $$ dx \cdot dy $$를 $$ dr \cdot d\theta $$로 변환하면 그 변환 term은 얼만큼 곱해주어야 할까요? 아래 그림을 통하여 직교 좌표계에서의 미소 면적과 극 좌표계에서의 `미소 면적`의 차이를 알아보겠습니다.

<br>
<center><img src="../assets/img/math/pb/about_gaussian/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 직교 좌표계에서는 $$ x, y $$의 변화량 $$ dx, dy $$에 의해 증가한 미소 면적은 직사각형으로 $$ dx \cdot dy $$ 입니다.
- 반면 극 좌표계에서 $$ r, \theta $$의 변화량 $$ dr, d\theta $$에 의해 증가한 영역은 $$ r \cdot dr \cdot d\theta $$ 입니다.
- (먼저 추상적으로 설명해 보면) 다음과 같은 적분의 성질을 보면 적분 구간을 근사(approximation)하여 면적을 구하는 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/math/pb/about_gaussian/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림의 왼쪽 그림을 보면 곡선은 함수 $$ r(\theta) $$를 따르고 $$ \theta $$의 범위는 $$ [a, b] $$입니다. 이 때, 면적 $$ R $$을 한번에 구하기 어렵기 때문에 다음 식을 통해서 근사화 하여 구할 수 있습니다.

<br>

$$ \frac{1}{2} \int_{a}^{b} r(\theta)^{2} d\theta $$

<br>

- 위 식을 유도해 보겠습니다. 오른쪽 그림과 같이 구간을 $$ n $$개로 나누고 각 구간을 $$ i =  1, 2, \cdots, n $$에서 $$ \theta_{i} $$이 각 구간의 중점이라고 하고 극에 중심을 두는 부채꼴을 만듭니다. 
- 이 때, 각 호의 반지름은 $$ r(\theta_{i}) $$, 중심각은 $$ \Delta \theta_{i} $$이면 호의 길이는 $$ r(\theta_{i})\Delta \theta $$가 되고 넓이는 $$ \frac{1}{2}r(\theta_{i})^{2} \Delta \theta $$가 됩니다. 따라서 총 넓이는 [리만 합](https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%A7%8C_%ED%95%A9)에 따라서 다음과 같이 정리됩니다.

<br>

$$ \sum_{i=1}^{n} \frac{1}{2} r(\theta_{i})^{2} \Delta \theta $$

<br>

- 리만 합의 정의에 따라 구간의 갯수 $$ n $$이 증가할수록 그 극한값은 넓이 $$ R $$에 가까워집니다.
- 이 성질을 이용하면 앞선 그림의 넓이 증가량을 $$ \color{blue}{r \cdot d\theta} \cdot \color{red}{dr} $$의 사각형 넓이로 근사하여 생각할 수 있습니다. 리만 합에 의해 $$ r, \theta $$의 값이 작은 단위로 나누어져서 합쳐진다면 실제 넓이에 가까워질 것이기 때문입니다. 
- 즉, discrete 한 케이스의 $$ \sum $$을 이용한 식의 구간을 무수히 많이 쪼개어서 합하게 되면 $$ \int $$ 형태의 합이 됩니다.
- 여기까지가 추상적이고 직관적인 설명이긴 합니다. 좀 더 구체적으로 알고 싶으면 아래 내용을 읽어보시면 도움이 됩니다. (넘어가셔도 됩니다.)

<br>

- 직교 좌표계에서는 미소 면적의 넓이를 $$ dA = dx \cdot dy $$로 표시하였습니다.
- 그리고 $$ x, y $$는 다음과 같이 $$ r, \theta $$로 나타내어 졌습니다.

<br>

$$ x = r \cos{(\theta)} $$

$$ y = r \sin{(\theta)} $$

<br>

- 여기서 $$ x, y $$ 에 대하여 각각 $$ r, \theta $$에의 변화량을 확인하기 위해 `자코비안`을 구해보면 다음과 같습니다.
- 아래 자코비안의 1행, 2행은 각각 $$ x, y, $$가 $$ r, \theta $$ 각각의 변화에 따라 얼만큼 변화를 가지는 지 나타냅니다.

<br>

$$ \frac{\partial(x, y)}{\partial(r, \theta)} = \begin{bmatrix} \partial x / \partial r & \partial x / \partial \theta \\ \partial y / \partial r & \partial y / \partial \theta \end{bmatrix} = \begin{bmatrix} \cos{(\theta)} & -r\sin{(\theta)} \\ \sin{(\theta)} & r\cos{(\theta)} \end{bmatrix} $$

- 변화의 **scale**을 구할 때, `determinant`를 사용할 수 있습니다. 지금과 같은 2차원에서 변화의 총량은 변화하였을 때의 `넓이`의 **scale**이 됩니다. 그러면 위에서 구한 자코비안의 determinant를 $$ J $$라고 나타내면 다음과 같습니다.

<br>

$$ J = \begin{vmatrix} \cos{(\theta)} & -r\sin{(\theta)} \\ \sin{(\theta)} & r\cos{(\theta)} \end{vmatrix} = r \cos^{2}{(\theta)} + r \sin^{2}{(\theta)} = r $$

<br>

- 즉, $$ r, \theta $$가 $$ dr, d\theta $$ 만큼 변할 때, 변화하는 양 scale은 $$ J = r $$ 이 됩니다.
- 따라서 직교 좌표계의 변화량과 극 좌표계의 변화량은 다음 관계를 가집니다.

<br>

$$ dA = dx \cdot dy = J \cdot dr \cdot d\theta = r \cdot dr \cdot d\theta $$

<br>
<center><img src="../assets/img/math/pb/about_gaussian/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다시 확인해 보면 극 좌표계에서 $$ r, \theta $$의 변화에 따른 변화량은 직관적인 설명과 자코비안을 통한 설명 모두 $$ r \cdot dr \cdot d\theta $$ 임을 확인할 수 있습니다. 

<br>
<center><img src="../assets/img/math/pb/about_gaussian/6.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 변화량의 scale은 $$ r $$ 입니다. 따라서 호의 반지름의 길이인 $$ r $$이 커질수록 미소 면적의 크기가 커짐을 확인할 수 있습니다.
- 지금 까지 확인한 내용인 직교 좌표계에서 극 좌표계로 변환을 통해 적분을 마무리 해보겠습니다.

<br>

$$ \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^{2} + y^{2})} \cdot dx \cdot dy = \int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^{2}} \cdot r \cdot dr \cdot d\theta $$

<br>

- 이제 식이 깔끔하게 정리되었으니 단순 치환 적분을 통하여 문제를 풀어보겠습니다.

<br>

$$ -r^{2} = u $$

$$ -2r dr = du $$

$$ r dr = -\frac{1}{2} du $$

$$ \begin{split} \int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^{2}} \cdot r \cdot dr \cdot d\theta &= \int_{0}^{2\pi}\int_{0}^{-\infty} e^{u} (-\frac{1}{2})du \cdot d\theta &= -\frac{1}{2} \int_{0}^{2\pi} [e^{u}]_{0}^{-\infty} d\theta &= \frac{1}{2} \int_{0}^{2\pi} d\theta = \pi \end{split}$$

$$ I^{2} = \pi = \Biggl( \int_{-\infty}^{\infty} e^{-x^{2}} dx \Biggr)^{2} $$

$$ \therefore \quad \int_{-\infty}^{\infty} e^{-x^{2}} dx = \sqrt{\pi} $$

<br><br>

## **가우시안 분포 공식 증명**

<br>

## **covariance와 zero-mean gaussian의 covariance**

<br>

- 가우시안 분포에서 공분산에 대하여 한번 알아보겠습니다.
- 다음과 같이 n 차원의 벡터 $$ x $$와 평균 벡터 $$ \mu $$가 있다고 가정해 보겠습니다.

<br>

$$ \boldsymbol x= \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} $$

<br>

- 벡터를 자세히 살펴보면 다음과 같습니다.

<br>

$$

\boldsymbol x= \begin{bmatrix} \color{blue}{x_1} \\ \color{red}{x_2} \end{bmatrix}=\begin{bmatrix}\color{blue}{x_{11} \\ x_{12}\\\vdots\\ x_{1h}}\\\color{red}{x_{21}\\x_{22}\\\vdots\\ x_{2k}}\end{bmatrix}\tag{$n \times 1$} 

$$

<br>

$$ \boldsymbol\mu= \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix} $$

<br>

- 이 때, covariance matrix는 다음과 같이 정의됩니다.

<br>

$$ 

\begin{bmatrix}
\Sigma_{\color{blue}{11}} & \Sigma_{\color{blue}{1}\color{red}{2}}\\
\Sigma_{\color{red}{2}\color{blue}{1}} & \Sigma_{\color{red}{22}}
\end{bmatrix} \tag {$n \times n$}

$$

<br>

- 이 때, 공분산도 자세히 살펴보면 다음과 같습니다.

<br>

$$

\Sigma_{\color{blue}{11}}=\begin{bmatrix}
\sigma^2({\color{blue}{x_{11}}}) & \text{cov}(\color{blue}{x_{11},x_{12}}) & \dots & \text{cov}(\color{blue}{x_{11},x_{1h}}) \\
\text{cov}(\color{blue}{x_{12},x_{11}}) & \sigma^2({\color{blue}{x_{12}}}) & \dots & \text{cov}(\color{blue}{x_{12},x_{1h}}) \\
\vdots & \vdots & & \vdots \\
\text{cov}(\color{blue}{x_{1h},x_{11}})  &  \text{cov}(\color{blue}{x_{1h},x_{12}}) &\dots& \sigma^2({\color{blue}{x_{1h}}})
\end{bmatrix} \tag{$h \times h$}

$$

<br>

$$

 \Sigma_{\color{blue}{1}\color{red}{2}}=
\begin{bmatrix}
\text{cov}({\color{blue}{x_{11}}},\color{red}{x_{21}}) & \text{cov}(\color{blue}{x_{11}},\color{red}{x_{22}}) & \dots & \text{cov}(\color{blue}{x_{11}},\color{red}{x_{2k}}) \\
\text{cov}({\color{blue}{x_{12}}},\color{red}{x_{21}}) & \text{cov}(\color{blue}{x_{12}},\color{red}{x_{22}}) & \dots & \text{cov}(\color{blue}{x_{12}},\color{red}{x_{2k}}) \\
\vdots & \vdots & & \vdots \\
\text{cov}({\color{blue}{x_{1h}}},\color{red}{x_{21}}) & \text{cov}(\color{blue}{x_{1h}},\color{red}{x_{22}}) & \dots & \text{cov}(\color{blue}{x_{1h}},\color{red}{x_{2k}})
\end{bmatrix}\tag{$h \times k$}

$$

<br>

$$

 \Sigma_{\color{red}{2}\color{blue}{1}}=
\begin{bmatrix}
\text{cov}({\color{red}{x_{21}}},\color{blue}{x_{11}}) & \text{cov}(\color{red}{x_{21}},\color{blue}{x_{12}}) & \dots & \text{cov}(\color{red}{x_{21}},\color{blue}{x_{1h}}) \\\text{cov}({\color{red}{x_{22}}},\color{blue}{x_{11}}) & \text{cov}(\color{red}{x_{22}},\color{blue}{x_{12}}) & \dots & \text{cov}(\color{red}{x_{22}},\color{blue}{x_{1h}}) \\
\vdots & \vdots & & \vdots \\
\text{cov}({\color{red}{x_{2k}}},\color{blue}{x_{11}}) & \text{cov}(\color{red}{x_{2k}},\color{blue}{x_{12}}) & \dots & \text{cov}(\color{red}{x_{2k}},\color{blue}{x_{1h}})
\end{bmatrix}\tag{$k \times h$}

$$

<br>

$$

\Sigma_{\color{red}{22}}=\begin{bmatrix}
\sigma^2({\color{red}{x_{21}}}) & \text{cov}(\color{red}{x_{21},x_{22}}) & \dots & \text{cov}(\color{red}{x_{21},x_{2k}}) \\
\text{cov}(\color{red}{x_{22},x_{21}}) & \sigma^2({\color{red}{x_{22}}}) & \dots & \text{cov}(\color{red}{x_{22},x_{2k}}) \\
\vdots & \vdots & & \vdots \\
\text{cov}(\color{red}{x_{2k},x_{21}})  &  \text{cov}(\color{red}{x_{2k},x_{22}}) &\dots& \sigma^2({\color{red}{x_{2k}}})
\end{bmatrix} \tag{$k \times k$}

$$

<br>

- 일단 covariance에 대하여 알아보았는데, 먼저 확인해야 하는 것은 행과 열의 인덱스가 같은 부분은 $$ \text{cov} $$가 아닌 분산의 형태인 $$ \sigma^{2} $$으로 나타나 있는가 입니다.
- 만약 $$ \text{cov} $$로 나타낸다면 예를 들면 $$ \text{cov}{x_{11}, x_{11}} $$이 됩니다.
- 여기서  $$ x_{i}, x_{j} $$가 벡터일 때, $$ \text{cov}{x_{i}, x_{j}} $$를 풀어보면 다음과 같습니다.

<br>

$$ \text{cov}({x_{i}, x_{j}}) = E[ (x_{i} - \mu_{i})(x_{j} - \mu_{j})] $$

<br>

- 따라서 $$ \text{cov}(x_{11}, x_{11}) =  E[(x_{11} - \mu_{11})^{2}] = \sigma^{2}(x_{11}) $$이 됩니다.

<br>

- 그러면 본론으로 들어가서 `zero-mean gaussian distribution`에 대하여 알아보겠습니다.
- 대각 성분은 앞에서 다룬 것 처럼 $$ \sigma^{2}(x_{ii}) $$ 형태가 됩니다.
- zero-mean gaussian에서는 **대각 성분이 아닌 경우에는 모두 0**이 됩니다.
- 대각 성분이 아닌 경우를 한번 살펴보면 다음과 같습니다.

<br>

$$ \text{cov}({x_{i}, x_{j}}) = E[ (x_{i} - \mu_{i})(x_{j} - \mu_{j}) ] = E[ (x_{i})(x_{j}) ]  = E[x_{i}]E[x_{j}]  = \mu_{i}\mu_{j} = 0 $$

<br>

- zero-mean gaussian 이기 때문에 $$ \mu $$는 모두 0이 되고 $$ E(XY) = E(X)E(Y) $$이기 때문에 분리가 됩니다. 분리한 각각의 값이 또 평균이기 때문에 각 평균은 0이되어 최종적으로 $$ \text{cov}{x_{i}, x_{j}} $$은 0이 됩니다.
- 따라서 `zero-mean gaussian distribution`에서 **대각 성분은 분산이 되고 그 이외 성분은 모두 0이 됩니다.**


$$

\Sigma_{}=\begin{bmatrix}
\sigma^2({\color{blue}{x_{1}}}) & 0 & \dots & 0 \\
0 & \sigma^2({\color{blue}{x_{2}}}) & \dots & 0 \\
\vdots & \vdots & & \vdots \\
0 & 0&\dots& \sigma^2({\color{blue}{x_{m}}})
\end{bmatrix} \tag{$h \times h$}

$$