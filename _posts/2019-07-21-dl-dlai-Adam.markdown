---
layout: post
title: Adam Optimization Algorithm
date: 2019-07-21 00:00:00
img: dl/dlai/dlai.png
categories: [dl-dlai] 
tags: [python, deep learning, optimization, Adam] # add tag
---

- 이전 글 : [RMSProp](https://gaussian37.github.io/dl-dlai-RMSProp/)

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/JXQT_vxqwIs" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

- 오늘 다룰 `Adam optimization` 방법은 모멘텀과 RMSProp 방법을 섞은 방법입니다.
    - Adam은 `Adaptive moment estimation`을 줄인 말입니다.
- Adam은 현재 다양한 딥러닝에서 많이 쓰이고 있고 또한 많이 사용되어 그 성능이 이미 증명이 된 상태입니다. 즉, Adam 알고리즘을 기본적으로 사용하면 좋은 성능을 보일 수 있으니 고민할 필요 없이 사용하면 됩니다.
- 그러면 Adam 알고리즘이 어떻게 동작하는지 알아보겠습니다.

<center><img src="../assets/img/dl/dlai/adam_optimization_algorithm/1.PNG" alt="Drawing" style="width: 800px;"/></center>

<br>

- Adam 알고리즘에서 사실 새로 배워야 하는 개념은 없습니다. 필요한 개념은 모두 [모멘텀](https://gaussian37.github.io/dl-dlai-gradient_descent_with_momentum/)과 [RMSProp](https://gaussian37.github.io/dl-dlai-RMSProp/)에서 모두 배웠습니다.
- 먼저 $$ v_{dw}, v_{db}, s_{dw}, s_{db} $$를 0으로 초기화 합니다. 이 때 사용되는 $$ v $$는 모멘텀에서 가져온 변수이고 $$ s $$는 RMSProp에서 가져온 변수입니다.
- 모멘텀과 RMSProp에서 다룬 것과 같이 현재 배치(미니 배치)를 대상으로 $$ dw, db $$를 구합니다.
- 그 다음 모멘텀을 구하는 과정과 똑같이 $$ v_{dw} = \beta_{1}*v_{dw} + (1-\beta_{1})dw, v_{db} = \beta_{1}*v_{db} + (1-\beta_{1})db $$를 구합니다.
    - 즉, 이 식은 하이퍼파라미터 $$ \beta_{1}$$을 사용한 모멘텀 업데이트입니다. 
- 추가적으로 RMSProp을 구하는 과정과 똑같이 $$ s_{dw} = \beta_{2}*s_{dw} + (1-\beta_{2})dw^{2}, s_{db} = \beta_{2}*s_{db} + (1-\beta_{2})db^{2} $$를 구합니다.
    - 즉, 이 식은 하이퍼파라미터 $$ \beta_{2}$$을 사용한 RMSProp 업데이트입니다.
- 그리고 전형적인 Adam 구현에서는 편향 보정(bias correction)을 적용합니다.
- 따라서 모멘텀 항은 $$ v_{dw}^{corrected} = v_{dw} / (1-\beta_{1}^{t}), v_{db}^{corrected} = v_{db} / (1-\beta_{1}^{t}) $$가 됩니다.
- 그리고 RMSProp 항은 $$ s_{dw}^{corrected} = s_{dw} / (1-\beta_{2}^{t}), s_{db}^{corrected} = s_{db} / (1-\beta_{2}^{t}) $$가 됩니다.

<br>

- 따라서 최종적으로 업데이트 되는 식은 $$ w = w - \alpha * \frac{ v_{dw}^{corrected} }{ \sqrt{s_{dw}^corrected} + \epsilon }, b = b - \alpha * \frac{ v_{db}^{corrected} }{ \sqrt{s_{db}^corrected} + \epsilon } $$이 됩니다.

<br>

- 위 식에서 알 수 있듯이 Adam은 RMSProp과 모멘텀에서 사용되는 그래디언트 디센트 효과를 모두 누릴 수 있습니다. 

<center><img src="../assets/img/dl/dlai/adam_optimization_algorithm/2.PNG" alt="Drawing" style="width: 800px;"/></center>

<br>

- Adam은 위의 식에서 볼 수 있듯이 하이퍼파라미터가 4개가 됩니다. $$ \alpha, \beta_{1}, \beta_{2}, \epsilon $$ 입니다.
- 여기서 $$ \alpha $$는 매우 중요하고 튜닝될 필요가 있으므로 다양한 값을 시도해서 잘 맞는 값을 찾아야합니다.
- 그리고 $$ \beta_{1} $$은 보통 0.9를 사용합니다. 이것은 $$ dw, db $$의 지수 가중 평균에 사용되는 값이지요.
- 그리고 $$ \beta_{2} $$은 0.999를 사용하길 권장합니다. (저자의 추천사항? 입니다.) 이것은 $$ dw^{2}, db^{2} $$의 지수가중평균에 사용되는 값입니다.
- 마지막으로 $$ \epsilon $$은 $$ 10^{-8} $$ 값을 사용하길 권장합니다. (이것 또한 저자의 추천입니다.)
- 수정해야할 하이퍼파라미터가 많지만 보통은 $$ \beta_{1}, \beta_{2}, \epsilon$$은 고정값으로 두고 $$ \alpha $$만 변경하여 학습을 하는 것이 일반적입니다.

<br>

- 다음 글 : [Learning Rate Decay](https://gaussian37.github.io/dl-dlai-learning_rate_decay/)
