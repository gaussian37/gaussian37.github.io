---
layout: post
title: Q3) Batch 사이즈는 중요한가?
date: 2018-12-14 00:00:00
img: interview/datascience/likelihood/ds_question.jpg
categories: [interview-datascience] 
tags: [interview, datascience, likelihood] # add tag
---

면접에서 Batch 사이즈는 중요한가? 라고 물어보면 어떻게 대답할까요?

저라면 아래와 같이 대답할 것 같습니다. 물론 정답은 아닐거 같지만요... 좋은 답 있으면 댓글로 남겨 주세요~

---

Batch 사이즈는 학습할 때 영향을 많이 준다고 생각합니다.
Batch 사이즈를 선정할 때에는 너무 크지도 않고 너무 작지도 않게 선정을 해야 합니다.
개인적으로는 $$ 2^{N} $$ 단위로 증가시키고 128 부터 사용하고 있습니다.
데이터에 따라 적당한 Batch 사이즈가 다르기 때문에 시행착오를 통해 적절한 사이즈를 찾아야 합니다.
Batch 사이즈가 너무 작으면 학습이 오래 걸리게 될 뿐 아니라 BatchNormalization을 사용할 때에도
각 Batch의 평균과 표준편차가 데이터의 분포를 잘못 나타낼 수 있고 Batch 사이즈가 너무 커버리면 
generalization performance 문제가 발생하게 됩니다. 이럴 때에는 gradient가 문제이기 때문에
optimizer 까지 고려하여 Batch 사이즈 문제를 해결하는 것이 좋습니다.