---
layout: post
title: Q2) Sigmoid 보다 ReLu를 많이 쓰는 이유?
date: 2018-11-10 18:43:00
img: interview/datascience/likelihood/ds_question.jpg
categories: [interview-datascience] 
tags: [interview, datascience, likelihood] # add tag
---

인터뷰 문제로 가장 흔하게 물어 볼수 있는 Sigmoid와 ReLu에 대해서도 대답할 정도만 알아 봅시다.

### Non-Linearity라는 말의 의미와 그 필요성은?

데이터에 대하여 Classification 또는 Regression 등을 할 때, 데이터의 분포에 맞게 boundary를 잘 그려줄 수 있다면
좋은 성능을 가진 모델링을 했다고 할 수 있습니다. 데이터의 복잡도가 높아지고 차원이 높아지게 되면 
데이터의 분포는 단순히 선형 형태가 아닌 비 선형 형태(Non-Linearity)를 가지게 됩니다.
이러한 데이터 분포를 가지는 경우 단순히 1차식의 Linear한 형태로는 데이터를 표현할 수 없기 때문에
Non-Linearity가 중요하게 됩니다.    

### ReLU로 어떻게 곡선 함수를 근사하나?

Multi-layer의 activation으로 ReLu를 사용하게 되면 단순히 Linear 한 형태에서 더 나아가 Linear 한
부분 부분의 결합의 합성 함수가 만들어 지게 됩니다. 이 Linear한 결합 구간 구간을 최종적으로 보았을 때,
최종적으로 Non-linearity 한 성질을 가지게 됩니다.
학습 과정에서 BackPropagation 할 때에, 이 모델은 데이터에 적합하도록 fitting이 되게 됩니다.
ReLu의 장점인 gradient가 출력층과 멀리 있는 layer 까지 전달된다는 성질로 인하여 데이터에 적합하도록
fitting이 잘되게 되고 이것으로 곡선 함수에 근사하도록 만들어 지게 됩니다.

### ReLU의 문제점은?

ReLU의 문제점은 입력값이 0보다 작을 때, 함수 미분값이 0이 되는 약점이 있습니다.

### Bias는 왜 있는걸까?

기하학적으로 생각해 보면 Bias는 모델이 데이터에 잘 fitting하게 하기 위하여 평행 이동하는 역할을 합니다.
데이터를 2차원으로 표현하였을 때, 모든 데이터가 원점기준에 분포해 있지는 않습니다. Bias 를 이용하여 모델이
평면 상에서 이동할 수 있도록 하고, 이 Bias 또한 학습하도록 합니다.

### Sigmoid의 문제점

첫째, 입력 값이 일정 범위의 safety zone을 넘어가게 되면 0 또는 1로 수렴하게 되고 gradient(경사값) 또한 0으로 수렴해 버리게 되어
학습이 제대로 안됩니다. 둘째, sigmoid function의 범위가 0 ~ 1 입니다. 이 때문에 output의 중앙값이 0이 아니게 됩니다. 
0을 기준으로 데이터가 분포하게 되었을 때가 이상적인데 sigmoid function은 그 분포 구간을 만족하지 못합니다.
셋째, Relu와 비교해 보았을 때, exp() 연산에 많은 cost가 듭니다.