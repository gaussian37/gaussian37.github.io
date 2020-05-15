---
layout: post
title: 가우시안 관련
date: 2019-11-17 00:00:00
img: math/pb/gaussian/gaussian.png
categories: [math-pb] 
tags: [gaussian, 가우시안] # add tag
---

<br>

- 공학에서 가우시안은 너무나 중요한 개념입니다. 공부하면서 종종 쓰였단 가우시안의 개념들을 정리해 놓는 글입니다.

<br>

## **목차**

<br>

- ### 가우스 적분 증명
- ### 가우스 공식 증명
- ### covariance와 zero-mean gaussian의 covariance

<br><br>

## **가우스 적분 증명**

<br>

- `가우스 적분`의 증명에 대하여 다루어 보도록 하겠습니다. 가우스 적분의 식은 다음과 같습니다.

<br>

$$ \int_{-\infty}^{\infty} e^{-x^{2}} dx = \sqrt{\pi} $$ 

<br>

- 가우스 적분의 증명을 이용하면 가우스 분포 공식의 유도에도 사용할 수 있기 때문에, 가우스 적분의 증명을 어떻게 하는 지 알아보도록 하겠습니다.

<br>

$$ I = \int_{-\infty}^{\infty} e^{-x^{2}} dx $$

$$ I^{2} = \Biggl( \int_{-\infty}^{\infty} e^{-x^{2}} dx \Biggr) = \int_{-\infty}^{\infty} e^{-x^{2}} dx \int_{-\infty}^{\infty} e^{-y^{2}} dy \\ \because \text{variable x is so called dummy variable. For this reason, it's possible } x \to y  $$


<br>

## **가우스 공식 증명**

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