---
layout: post
title: Q4) Dimension 이란 무엇인가?
date: 2018-12-16 00:00:00
img: interview/datascience/likelihood/ds_question.jpg
categories: [ml-question] 
tags: [interview, datascience, dimension, feature] # add tag
---

Dimension이란 무엇일까요?

---

수학에서 Dimension 즉, 차원이라고 하면 1차원은 직선에 나타낼 수 있고 2차원은 평면에 나타낼 수 있습니다.
즉, x 축, y축 등으로 나타내는 축을 간단하게 Dimension 이라고 할 수 있었습니다.

머신 러닝에서 말하는 이 Dimension 도 같습니다. 데이터를 표현하는 데 필요한 축을 Dimension이라고 할 수 있습니다.
즉, Feature의 갯수를 Dimension의 갯수라고 볼 수 있습니다. 예를 들어 `정규 분포`를 나타낼 때에는 `평균`과 `표준 편차` 라는 두 변수가 있습니다.
따라서 Feature의 갯수가 2개 이므로 2-dimension에 패턴을 나타낼 수 있습니다.

반면 이미지의 경우는 각 픽셀에 해당하는 값이 Feature 라고 볼 수 있습니다.
Toy data인 MNIST의 경우 28 x 28 x 1의 크기를 가집니다. 총 784개의 픽셀을 가지고 있습니다.
이 때에는 784개의 Feature를 가지고 Dimension의 크기도 784개라고 할 수 있습니다.

추가적으로 이렇게 큰 Dimension은 사람이 확인하기가 어렵습니다. 따라서 가장 중요하다고 생각되는
몇 개의 Dimension (2, 3개)로 사람이 눈으로 확인할 수 있는 형태로 만들면 데이터의 분포(평면 또는 입체 형태로)를 확인할 수 있습니다.
이것을 Dimension reduction (차원 축소) 라고 합니다.