---
layout: post
title: AutoEncoder의 모든것 (2. Manifold Learning)
date: 2019-02-25 00:00:00
img: gan/concept/autoencoder1/autoencoder.png
categories: [gan-concept] 
tags: [deep learning, autoencoder] # add tag
---

+ 이 글은 오토인코더의 모든것 강의를 보고 요약한 글입니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/o_peo6U7IRM" frameborder="0" allowfullscreen="true" width="400px" height="800px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-1.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 글에서는 `manifold learning`이 무엇인지 직관적으로 한번 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-2.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이미지 데이터의 경우 상당히 큰 차원의 데이터를 가지고 있습니다. 예를 들면 256 x 256 사이즈의 RGB 데이터 라면 196,608(256 x 256 x 3)의 차원을 가지고 있습니다. 
- **manifold learning**의 한가지 역할 중 하나는 차원 축소 인데, 굉장히 큰 차원을 저차원으로 줄일 수 있는 역할을 합니다.
- 위 슬라이드의 예를 들어 고차원(시각화를 위해 3차원으로 표현되어 있습니다.)을 저차원(2차원)으로 매핑 시키는 것을 한번 살펴보겠습니다.
- 그러면 먼저 **매니폴드**에 대한 정의를 먼저 살펴보도록 하겠습니다.
- 공간상에 데이터들을 표현하면 각 데이터들은 점의 형태로 찍혀질 것입니다. **매니폴드**는 이런 점들을 최대한 에러 없이 아우를 수 있는 **서브 스페이스**라고 정의할 수 있습니다. 
- 이 **매니폴드**를 잘 찾는 것이 **매니폴드 러닝**이라고 할 수 있고 잘 찾은 **매니폴드**를 저차원에 프로젝션 하면 저차원으로 차원 축소를 할 수 있는 것입니다.

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-3.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>  

- **미니폴드 러닝**은 간단하게 높은 차원의 데이터를 저 차원으로 잘 줄이고 싶은 것입니다.
    - 특히 원래 데이터의 특성을 잘 유지하면서 차원을 축소해야 하는 작업니다.
- 이렇게 차원을 축소한 결과는 크게 4가지 용도로 사용될 수 있습니다.
    - Data compression
    - Data visualization
    - Curse of dimensionality 개선
    - Discovering most important features
- 먼저 Data compression과 관련된 내용부터 살펴보겠습니다.    
    
<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-4.jpg" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 위 논문은 데이터 압축과 관련된 논문입니다.
- 사진 압축에 많이 사용되는 것 중 하나가 **JPEG**인데 **JPEG**과 비교하여 데이터 압축이 잘 되었다는 내용의 논문입니다.
- 입력의 이미지를 차원 축소 하여 저차원의 데이터(저용량의 데이터)를 가지고 있으면 다시 복원하여 사용할 수가 있습니다.

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-5.jpg" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 두번째 사용처인 **Visualization**은 위 슬라이드의 예제와 같습니다.
- 위 슬라이드 예제는 `t-SNE` 방법으로 가장 많이 사용하는 Visualization 방법 중 하나입니다.

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-6.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- 그 다음이 `차원의 저주`입니다.
- 차원의 저주는 차원이 높아질수록 공간이 커지는 반면에 데이터의 수는 고정되어 있어 데이터의 밀도가 희박해 지는 문제를 뜻합니다.
- 예를 들어 1차원 총 공간이 10이고 데이터가 8이였으면 전체 공간에 80%만큼 데이터가 채워져 있었지만 2차원으로 늘어나면 전체 공간 100에서 8이므로 8% 공간만 차지하게 되고 3차원, 4차원으로 늘어날수록 그 비율은 더 줄어 들게 되는 문제입니다.
    - 이렇게 전체 공간에 비하여 데이터가 너무 작아지면 분석 자체가 어려워져서 성능이 나빠지게 됩니다. 

<br>
<center><img src="../assets/img/gan/concept/autoencoder2/2-6.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위와 같이 고차원의 밀도는 낮지만 이들의 집합을 포함하는 저차원의 매니폴드를 찾으면 이 매니폴드 상에서는 데이터의 밀도가 높게 나타나 집니다.

    