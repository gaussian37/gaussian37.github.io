---
layout: post
title: PCA(Principal Component Analysis) (Andrew Ng)
date: 2019-04-02 03:49:00
img: ml/concept/about-pca/pca.png
categories: [ml-concept] 
tags: [machine learning, pca] # add tag
---

출처 : Andrew Ng 강의

### Dimensionality Reduction

+ 이번 글에서는 `PCA(Principal Component Analysis)`에 대하여 알아보도록 하겠습니다.
+ PCA에 대하여 자세히 알아보기 전에 unsupervised learning의 한 종류에 해당하는 `dimensionality reduction`에 대하여 먼저 살펴보도록 하겠습니다.
+ 그 중 첫번째로 살펴볼 `dimensionality reduction`의 역할은 `compressing the data` 입니다.

<img src="../assets/img/ml/concept/about-pca/1.png" alt="Drawing" style="width: 600px;"/>

+ 만약 데이터를 표현하는 feature가 여러개가 있다고 가정해 보겠습니다. 위 슬라이드에서는 간단하게 표현하기 위해서 feature가 2개 있다고 가정하겠습니다.
+ 위 슬라이드의 2개의 feature는 사실 중복된 feature라고 볼 수 있습니다. 둘 다 길이를 나타내기 때문입니다.
    + 만약 수십, 수백개의 feature가 있다고 하더라도 그 중에서는 여러개의 중복된 feature가 분명히 있을 수 있습니다.
+ 따라서 중복된 feature를 하나로 줄이기 위하여 현재 분포된 데이터 상에서 데이터를 표현하는 파란색`선`을 그려보면 위와 같습니다.
+ 이 새로운 선은 새로운 feature라고 생각할 수 있습니다. 이 새로운 선으로 데이터를 표현하면 2개의 feature인 $$ x^{(1)}, x^{(2)} $$ 대신에 새로운 feature인 $$ z^{(1)} $$으로 대신 표현할 수 있습니다.  

+ 그러면 원래 데이터에 해당하는 $$ x^{(i)} \in \mathbb R^{2} $$는 $$ z^{(i)} \in \mathbb R^{(1)} $$로 대응되게 됩니다.
         
<img src="../assets/img/ml/concept/about-pca/2.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드에서는 조금 더 차원을 확장해서 3차원을 2차원으로 축소해 보도록 하겠습니다. 즉, feature의 갯수가 3개에서 2개로 축소하는 것입니다.
+ 방법은 앞에 방법과 동일합니다. 기존의 feature인 $$ x^{(1)}, x^{(2)}, x^{(3)} $$을 이용하여 표현한 공간에 새로운 feature인 $$ z^{(1)}, z^{(2)} $$ 만을 이용하여 데이터를 표현합니다.
+ 즉, 공간에서 그려졌던 데이터가 평면에 표현된 것을 볼 수 있습니다.

<br>

+ 지금까지 살펴본 것을 이용하여 `dimensionality reduction`에 대하여 정의해 보면
+ 만약 데이터가 $$ \{x^{(1)}, x^{(2)}, ... , x^{(m)} \}, x^{(i)} \in \mathbb R^{n} $$이 있을 때,
+ lower dimensional dataset은  $$ \{z^{(1)}, z^{(2)}, ... , z^{(m)} \}, z^{(i)} \in \mathbb R^{k}, k \le n $$ 입니다.

<br>

+ 추가적으로 `dimensionality reduction`의 또 다른 역할 중 하나인 `data visualization`에 대하여 알아보도록 하겠습니다.
+ 먼저 데이터를 시각화하면 어떤 장점이 있을까요? 간단하게 말하면 데이터에 대한 분포를 눈으로 쉽게 확인 할 수 있고 데이터에 대한 이해도 증가하게 됩니다.

<img src="../assets/img/ml/concept/about-pca/3.PNG" alt="Drawing" style="width: 600px;"/>

+ 만약 위 슬라이드 처럼 여러개의 feature들이 있다면(GDP, Life, Poverty, ...) 각 나라의 특성을 한눈에 확인하기가 어렵습니다.
    + 즉, 데이터의 특성에 대한 이해도가 떨어질 수 있습니다.

<img src="../assets/img/ml/concept/about-pca/4.PNG" alt="Drawing" style="width: 600px;"/>

+ 반면 위 슬라이드 처럼 단 2개의 새로운 feature로 데이터를 표현한다면 새로운 공간에 데이터들을 projection할 수 있고, 클래스들 간의 분포등을 쉽게 볼 수 있습니다.
   
<img src="../assets/img/ml/concept/about-pca/5.PNG" alt="Drawing" style="width: 600px;"/>

+ 예를 들어 새로운 feature가 뜻하는 것이 위 처럼 country size/GDP 처럼 특정 의미를 지니게도 만들 수 있습니다.
+ 하지만 위 처럼 인위적으로 feature를 새로 만들어서 feature의 갯수를 줄인다기 보다는 좀 더 수학적으로 데이터들을 잘 표현할 수 있도록 대표하는 feature를 만드는 방법을 사용할 예정입니다.

<br><br>

### PCA(Principal Component Analysis)

+ 대표적인 `dimensionality reduction` 방법중의 하나인 PCA(Principal Component Analysis)에 대하여 알아보도록 하겠습니다.

<img src="../assets/img/ml/concept/about-pca/6.PNG" alt="Drawing" style="width: 600px;"/>

+ 위의 슬라이드와 같이 검은색 좌표들 처럼 데이터가 분포가 되어 있다고 가정해 봅시다. 
+ 위 슬라이드의 데이터는 2차원 좌표에 분포해 있으므로 차원을 줄이려면 1차원 즉, 선 하나에 데이터를 위치시켜야 합니다.
+ 따라서 데이터들을 빨간색 선에 위치 시키도록 해보겠습니다.
+ 그러면 여기서 빨간색 선을 어떻게 구할 수 있을까요? 방법중의 하나는 검은색 점들을 정사영시켰을 때의 좌표들을 선으로 잇는 방법이 있습니다.
+ 이 방법으로 저차원의 공간(여기서는 1차원이므로 선)을 구하면 여러가지가 나올 수 있습니다.
+ 이 때 정사영시킨 거리(`파란색선`)의 합이 최소가 되도록 하는 선을 구하면 모든 데이터들을 가장 잘 표현할 수 있는 새로운 차원의 축을 만들 수 있습니다.
    + 참고로 `파란색선`을 `projection error` 라고 합니다.

<img src="../assets/img/ml/concept/about-pca/7.PNG" alt="Drawing" style="width: 600px;"/>    
    
+ 예를 들어 위의 자주색 선의 `projection error`를 보면 빨간색 선에 비해서 상당히 큰 것을 볼 수 있습니다.
    + 이런 이유로 자주색 선 같은 정사영의 집합으로 이루어진 새로운 축 보다는 빨간색 선 같은 축이 선택됨을 알 수 있습니다.  
+ 위 방법을 이용하여 데이터를 정사영시킬 새로운 공간(선, 면등)을 찾는 방법을 `PCA` 라고 합니다.
      
+ `PCA`에 대한 내용을 정리하면 앞에서 살펴본 바와 같이 2차원 데이터를 1차원으로 차원 축소 시킬 때, 데이터들을 1차원으로 정사영시킬 벡터를 찾았습니다.
    + 이 때, 벡터는 projection error를 최소로 만들 수 있는 벡터라는 점을 이용하여 찾았습니다.
+ 이것을 일반화 시켜서 n차원 데이터를 k차원 데이터로 차원 축소를 시킨다면 어떻게 할 수 있을까요?
+ 이 때에는 k개의 벡터를 찾아야 합니다. 이 때 벡터를 $$ u^{(1)}, u^{(2)}, ..., u^{(k)} $$ 라고 할 수 있습니다.
+ 이 때 찾은 k개의 벡터를 이용하여 공간을 만들고 데이터들을 k개의 벡터들로 만든 공간안에 정사영 시킬 수 있습니다.
    + 같은 원리로 k개의 벡터들은 데이터들을 정사영시켰을 때 projection error가 최소가 되는 벡터들 입니다.
    + 그리고 선형대수학의 이론을 빌려서 표현하면 k개의 벡터로 표현한 공간은 $$ span\{ u^{(1)}, u^{(2)}, ..., u^{(k)} \} $$ 가 됩니다.       

<br>

+ 간혹 PCA를 이용하는 것과 linear regression하는 것을 헷갈릴 수도 있습니다. 하지만 큰 차이가 있습니다.
      
<img src="../assets/img/ml/concept/about-pca/8.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드를 보면 왼쪽은 linear regression이고 오른쪽은 PCA입니다.
    + 왼쪽을 보면 데이터와 선의 점간의 거리 차를 계산하는 것입니다.
    + 오른쪽을 보면 데이터들을 정사영 시킨 것입니다.
+ 또한 왼쪽의 linear regression에서는 정답 데이터가 필요한 반면 오른쪽의 PCA에서는 정답 데이터가 필요없는 차이점도 있습니다.
    + 즉 linear regression은 supervised learning 방법으로 prediction을 하는 것인 반면
    + PCA는 projection을 할 새로운 차원을 만들어내는 unsupervised learning 방법입니다.

<br>

+ 그러면 PCA를 구하려면 어떻게 구해야 하는지 좀 더 수식적으로 알아보도록 하겠습니다.
+ 먼저 PCA를 하기 전에 데이터 전처리를 해주어야 합니다. 즉 데이터에 대하여 `normalization`을 합니다.

<img src="../assets/img/ml/concept/about-pca/9.PNG" alt="Drawing" style="width: 600px;"/>

+ `normalization`을 해주는 이유는 데이터들이 유사한 범위의 값을 가지도록 하기 위해서 입니다.

<img src="../assets/img/ml/concept/about-pca/10.PNG" alt="Drawing" style="width: 600px;"/>

+ PCA 연산을 하면 두가지 결과를 얻을 수 있습니다.
    + 위 슬라이드에서 표현되는 새로운 공간 축인 $$ u^{(i)} $$ 입니다.
    + 그리고 새로운 공간 축에 대응되는 새로운 데이터의 값인 $$ z^{(i)} $$ 입니다.
    
<img src="../assets/img/ml/concept/about-pca/11.PNG" alt="Drawing" style="width: 600px;"/>

+ 다음으로 PCA를 하려면 n차원의 데이터를 k차원으로 줄여야 하는데 이 때 사용할 방법이 `공분산`과 `고유벡터`입니다.
+ 먼저 n차원 데이터 $$ x^{(i)} $$ 에 대한 `공분산`을 구해야 합니다.
    + 위 슬라이드에서 각 데이터 $$ x^{(i)} $$는 n차원이므로 (n x 1) 벡터로 표현할 수 있고 데이터의 갯수가 m개 이므로 m으로 평균을 만들어서 공분산을 구해줍니다.
+ 공분산을 이용해서 특이값 분해를 하면 `고유벡터`를 구할 수 있습니다.         
    + 특이값 분해(SVD)는 다음 링크를 참조하시기 바랍니다
        + https://gaussian37.github.io/math-la-linear-algebra-basic/
+ 이 때 행렬 $$ U $$는 (n x n)의 크기를 가지게 되고 여기서 k개를 선택하면 차원의 수를 k개로 줄일 수 있습니다.
+ 즉 정리하면,
    + 1 ) 공분산 행렬을 구합니다.
    + 2 ) 공분산 행렬의 고유벡터를 구합니다.
    + 3 ) 고유벡터에서 k개 만큼 선택하여 차원의 갯수를 k개로 줄입니다.
+ 이 방법이 `PCA`의 핵심입니다. 
 
<img src="../assets/img/ml/concept/about-pca/12.PNG" alt="Drawing" style="width: 600px;"/>

+ 좀 더 풀어서 살펴보면 행렬 $$ U $$ 는 위와 같이 (n x k)의 크기를 가지게 됩니다.
+ 새로운 공간에 값을 projection시킨 값 $$ z^{(i)} $$를 구하려면 $$ z^{(i)} = U^{T}x^{(i)} $$를 통하여 (k x 1) 크기의 벡터를 구할 수 있습니다. 

<img src="../assets/img/ml/concept/about-pca/13.PNG" alt="Drawing" style="width: 600px;"/>

+ 따라서 코드로 `PCA`를 표현하면 위와 같이 쉽게 나타낼 수 있습니다.