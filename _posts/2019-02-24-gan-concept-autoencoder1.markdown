---
layout: post
title: AutoEncoder의 모든것 (1)
date: 2019-02-24 00:00:00
img: dl/concept/autoencoder1/autoencoder.png
categories: [gan-concept] 
tags: [deep learning, autoencoder] # add tag
---

+ 출처 : https://www.youtube.com/watch?v=o_peo6U7IRM

+ 이 글은 오토인코더의 모든것 강의를 보고 요약한 글입니다.

<center><img src="../assets/img/gan/concept/autoencoder1/1-1.jpg" alt="Drawing" style="width: 800px;"/></center>

+ AutoEncoder와 관련된 키워드는 다음과 같습니다.
    + Unsupervised Learning
    + Representation learning = Efficient coding learning
    + Dimensionality reduction
    + Generative model learning
+ 여기서 가장 많이 알려진 키워드가 `Dimensionality reduction` 입니다.
    + 많은 사람들이 이 용도로 AutoEncoder를 사용하고 있습니다. 

<center><img src="../assets/img/gan/concept/autoencoder1/1-2.jpg" alt="Drawing" style="width: 800px;"/></center>

+ Nonlinear Dimensionality Reduction과 같은 용도로 사용되는 키워드는 위와 같습니다.
+ 이 때, `Feature Extraction`과 `Manifold learning`은 자주 사용되는 용어입니다.

<center><img src="../assets/img/gan/concept/autoencoder1/1-3.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 벤지오의 기술분류표를 보면 AutoEncoder는 Representation Learning에 해당합니다.

<center><img src="../assets/img/gan/concept/autoencoder1/1-4.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 이안 굿펠로우의 기술분류표를 보면 `Variational Autoencoder`는 `Maximum Likelihood`와 연관이 되어 있습니다.
+ 따라서 Maximum Likelihood에 대해서도 알아보도록 하겠습니다. 

<center><img src="../assets/img/gan/concept/autoencoder1/1-5.jpg" alt="Drawing" style="width: 800px;"/></center>

+ AE(AutoEncoder)는 입력과 출력이 같은 동일한 구조를 가지고 있으면 AE라고 부릅니다.
+ 이 때 관계된 키워드가 앞에서 언급한 바와 같이 크게 4가지가 있습니다.
+ **오토인코더를 학습**할 때,
    + `Unsupervised learning` : 학습 방식에 해당합니다.
    + `ML density estimation` : `Loss function`이 `Negative Maximum Likelihood`로 해석을 해서 Loss를 Minimize 하면 `Maximum Likelihood`가 됩니다.
+ **학습된 오토인코더**에서
    + `Manifold learning` : 인코더는 차원 축소 역할을 수행하고 (일반적으로 가운데 차원은 입력 보다 작음) 이것을 Manifold learning 이라고 합니다.
    + `Generative model learning` : manifold learning을 한 결과를 다시 입력 차원과 똑같이 복원을 하는 데 이것을 Generative model 이라고 합니다.   

<center><img src="../assets/img/gan/concept/autoencoder1/1-6.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 앞으로 다룰 내용은 위의 슬라이드에 있는 구성으로 진행할 것입니다.
+ 챕터 1, 2는 Variational AutoEncoder(VAE)를 이해하기 위한 `사전지식`으로 구성되어 있습니다.
    + 챕터1은 AE가 Maximum Likelihood(ML)과 관련이 있음을 설명합니다.
    + 챕터2는 AE가 Manifold learning에 많이 쓰이기 때문에 Manifold learning에 대한 정의에 대하여 알아보도록 하겠습니다.
+ 챕터 3에서는 `기본적인 AE`에 대하여 다루어 보도록 하겠습니다.
+ 챕터 4에서는 `VAE`에 대하여 다루어 보도록 하겠습니다.
+ 챕터 5에서는 `AE를 응용하는 방법`에 대하여 알아보도록 하겠습니다.

<center><img src="../assets/img/gan/concept/autoencoder1/1-7.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 챕터1을 시작해 보겠습니다. 챕터1에서는 AE는 ML하는 작업인 것을 이해하면 되겠습니다. 

<center><img src="../assets/img/gan/concept/autoencoder1/1-8.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 먼저 전통적인 머신러닝 방법부터 살펴보도록 하겠습니다. 
+ 머신러닝을 할 때에는 주어진 데이터가 있으면 그 데이터에서 필요한 정보를 뽑아내야 합니다. (얼굴이 있으면 이것은 얼굴이다? 또는 누구의 얼굴이다와 같은 추상적인 정보)
+ 따라서 먼저 학습을 하기 위한 `데이터`를 먼저 모아야 합니다. 특히 위의 슬라이드를 보면 $$ x, y $$ 데이터를 모아서 데이터 셋 $$ D $$를 구성하게 됩니다.
    + 이 데이터 구성은 supervised learning을 위한 방법입니다. 
+ 입력 데이터를 마련하였다면 그다음은 학습할 `모델`을 준비해야 합니다.
    + 즉, 주어진 데이터에서 추상적인 답을 뽑아낼 알고리즘을 정의해야 합니다. 
+ 모델을 정의한 다음에는 모델을 결정지을 `파라미터`를 학습하여 결정해야 합니다.
+ 이 때 학습을 하기 위해 `Loss function`을 정의해야 합니다. 
+ 정의한 `Loss function`을 이용하여 모든 데이터셋에 걸쳐서 예측값과 정답간의 차이를 가장 좁혀주는 파라미터를 찾습니다.
+ 학습이 완료되면 예측을 하고 성능을 테스트 합니다.  

<center><img src="../assets/img/gan/concept/autoencoder1/1-9.jpg" alt="Drawing" style="width: 800px;"/></center>

+ 딥러닝에서는 위에서 설명한 머신러닝 방법에서 모델 설정 방법이 조금 바뀌게 됩니다.
+ 딥러닝에서는 모델을 `Deep Neural Network`를 사용하는것으로 정의합니다. 
+ 그리고 딥러닝에서는 `Loss function`을 정의할 때 주로, `Mean Squared Error`, `Cross Entropy`등을 사용합니다.
    + 왜냐하면 `Back-propagation`을 하기 위해서 입니다.
    + `Back-propagation`을 통해서 딥 뉴럴 네트워크를 학습해야 하고 위에서 언급한 2가지 `Loss`가 효율적으로 사용될 수 있습니다.
        + 딥러닝에서는 뉴럴네트워크 구조로 인하여 `Back-propagation`을 통한 학습을 해야 하는데, 이 때 필요한 2가지 조건이 있습니다. 
        + 이 조건을 만족하는 `Loss`가 `MSE`와 `CE`입니다. 
            + 조건1 : 전체 데이터셋에 대한 Loss는 각각의 데이터의 Loss의 `합`과 같다.
            + 조건2 : Loss를 구성할 때, `네트워크의 출력값` 만을 사용한다. 

<center><img src="../assets/img/gan/concept/autoencoder1/1-10.jpg" alt="Drawing" style="width: 800px;"/></center>

<br>

+ 전체 학습 데이터 셋에 대한 `Loss`를 최소화 하는 파라미터를 찾는데 주로 사용하는 방법은 `Gradient Descent`입니다.
+ `Gradient Descent(GD)`는 기본적으로 iterative 한 방식입니다. 
    + `Closed form`이 아니므로 한번에 솔루션을 찾지는 못하고 여러번 시도 끝에 솔루션을 찾아가는 구조입니다. 
+ Iterative method에는 2가지가 결정되어야 합니다.
    + 어떻게 파라미터를 업데이트 할 것인가 ?
        + `GD`에서는 `Loss`값이 줄어들기만 하면 업데이트를 하는 전략을 취합니다.
    + 언제 업데이트를 멈출 것인가?
        + `GD`에서는 파라미터를 아무리 바꾸더라도 `Loss`가 줄어들지 않으면 업데이트를 멈춘다는 전략을 취합니다.
        
<center><img src="../assets/img/gan/concept/autoencoder1/1-11.jpg" alt="Drawing" style="width: 800px;"/></center>

<br>

- 그런데 한가지 문제점이 있습니다. 업데이트 해야 할 파라미터 $$ \theta $$의 차원이 굉장히 큰 경우에는 어떻게 $$ \theta $$를 변경해야 할까요?
- 즉, 어떤 방법으로 파라미터를 업데이트 해야 할 지에 대한 전략이 필요합니다.
- 앞의 전략(Loss가 줄어드는 방향으로 파라미터 업데이트)을 만족시키면서 파라미터를 변경하는 방법에 대하여 알아보겠습니다.
- 머신러닝, 딥러닝에서 사용하는 파라미터 업데이트 방법의 기본은 `taylor expansion`에 기초를 두고 있습니다.

<center><img src="../assets/img/gan/concept/autoencoder1/taylor.png" alt="Drawing" style="width: 600px;"/></center>

<br>

- taylor expasion의 정의는 위와 같습니다.
- 간단히 말하면 변경 전 정의역의 치역값에 정의역의 변경값과 1차 미분값의 곱, 정의역의 변경갑소가 2차미분값의 곱, ... 을 모두 더하면 변경 후 정의역의 치역값이 된다는 것입니다.
- 머신러닝에서는 이 taylor expansion의 방법에서 1차 미분값 까지만 사용해서 근사값을 추정하는 방식을 사용합니다.
    - 즉, $$ \theta $$를 바꾸었을 때의 Loss 값은 현재 Loss 값 + 그래디언트 * $$ \theta $$의 변경량이 됩니다.
    - taylor expansion에서는 어떤 값($$ x=a $$)을 기준으로 미분하였으므로 $$ x=a $$에서 다항근사를 하는 것이지만 그 주변까지도 어느 정도는 근사가 됩니다.
        - 이 근사가 되는 범위는 몇 차수 미분 까지 사용하는 것과 관계가 있습니다. 더 많은 차수를 사용할수록 더 넓은 지역을 작은 오차로 표현이 가능합니다.
        - 예를 들면 sin 함수에서 기울기가 1인 부분이 있습니다. 그 지점에서만 국소적으로 바라봤을 때는 sin(x) 함수와 y=x 함수가 똑같아 보입니다. 하지만 그 지점을 조금 더 벗어나면 sin(x)와 y=x값이 달라보이게 됩니다. 
        - 만약 근사값을 사용할 때, 1차 미분이 아니라 2차, 3차, ... 미분까지 사용한다면 y=x 함수식 보다 조금 더 복잡하지만 그 부근에서 y=x 일 때 보다 좀 더 넓은 범위로 sin(x)와 똑같아 보이는 근사식을 구할 수 있습니다. 
- 그러면 1차 미분값 까지만 사용한 근사식을 이용하여 Loss값의 변화량을 살펴보면 위 슬라이드와 같이 $$ triangle L \approx \nabla L * \triangle \theta $$가 됩니다.
    - 참고로 $$ \triangle $$ 은 변화량(델타)을 뜻하고 $$ \nabla $$는 그래디언트(미분값)을 뜻합니다.
- 만약 $$ \triangle \theta = -\eta\nabla L $$ 라고 한다면, $$ \triangle L = -\eta \Vert \nabla L \Vert < 0 $$ 이 됩니다.
    - 이 때, $$ \eta $$가 양수이고 흔히 아는 learning rate를 뜻합니다.
    - 위 식에 의해서 $$ \triangle L $$은 항상 음수가 되로 Loss는 줄어드는 방향으로 업데이트가 됩니다.
- 그러면 $$ \triangle \theta = -\eta\nabla L $$ 라는 가정은 왜 두었을까요? 앞에서 설명한것과 같이 Loss 값을 항상 줄이는 방향으로 만들기 위해서 입니다. (학습의 목적이지요)
- 즉 $$ \triangle \theta = -\nabla L $$이 되도록 하면 항상 Loss는 줄어들게 되고 여기에 $$ \eta $$ 값을 곱해주어서 학습이 잘 될수 있도록 $$ \triangle \theta $$ 값의 변경 량을 조절해 줍니다.
- 이 때 Learning rate $$ \eta $$를 작은 값을 사용하는 이유는 파라미터 값을 조금씩 바꾸기 위함입니다.
    - 파라미터 값을 조금씩 바꿔야 하는 이유는 Loss 함수에서 taylor expasion의 1차 미분항까지만 사용했기 때문에 아주 좁은 영역에서만 감소 방향이 정확하기 때문입니다.
    - 즉, 파라미터 변경량이 크면 근사를 잘못할 수 있기 때문에 learning rate를 이용하여 변경량을 적당히 줄여줍니다.
    - learning rate가 너무 커서 Loss가 발산하는 경우가 바로 approximation을 잘못한 경우입니다.

<center><img src="../assets/img/gan/concept/autoencoder1/1-12.jpg" alt="Drawing" style="width: 800px;"/></center>

<br>

- 

<center><img src="../assets/img/gan/concept/autoencoder1/1-13.jpg" alt="Drawing" style="width: 800px;"/></center>

<center><img src="../assets/img/gan/concept/autoencoder1/1-14.jpg" alt="Drawing" style="width: 800px;"/></center>

<center><img src="../assets/img/gan/concept/autoencoder1/1-15.jpg" alt="Drawing" style="width: 800px;"/></center>

        
    