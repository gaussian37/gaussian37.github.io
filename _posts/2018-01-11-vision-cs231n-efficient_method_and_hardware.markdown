---
layout: post
title: 15. Efficient Methods and Hardware for Deep Learning
date: 2018-01-11 15:00:00
img: vision/cs231n/15/0.png
categories: [vision-cs231n] 
tags: [cs231n, efficienct method and hardware] # add tag
---

<br>

[CS231n 강의 목록](https://gaussian37.github.io/vision-cs231n-table/)

<br>


<br>

## **목차**

<br>

- ### Summary

<br>


<br>

## **Summary**

<br>

- 딥러닝이 발전함에 따라 모델 사이즈와 연산량이 해마다 커지는 추세이며 이에 따른 문제점이 제기되고 있습니다.
    - ① `Model Size` : 모바일이나 자율주행자동차 등에 무선망으로 배포하기에는 모델 사이즈가 큰 문제가 있습니다.
    - ② `Speed` : Training time이 지나치게 길어질 수 있습니다.
    - ③ `Energy Efficiency` : 배터리 소모 문제 (모바일), 전기 비용 문제 (데이터 센터) 가 발생할 수 있습니다. 특히 이와 같은 경우 연산보다는 메모리 접근에 에너지 소모가 크므로 모델 크기를 줄이는 것이 중요합니다.

<br>

- 다음으로 숫자를 표현하는 데 사용하는 데이터 타입에 따른 배경 지식을 소개하겠습니다.

<br>
<center><img src="../assets/img/vision/cs231n/15/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 딥러닝 학습 시에 사용하는 대표적인 숫자 데이터 타입은 `FP32`와 `FP16`입니다. 이 타입을 구성하는 비트 정보는 위 그림과 같으며 Range와 Accuracy도 확인하시기 바랍니다.

<br>
<center><img src="../assets/img/vision/cs231n/15/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 도표는 숫자 표현 및 메모리 접근에 대한 비용을 나타냅니다. 특히 메모리 접근에 대한 비용이 굉장히 큰 것을 알 수 있습니다.
- 이를 통하여 모델 사이즈 감소 및 가벼운 타입의 숫자 자료형을 사용하는 것의 중요성을 알 수 있습니다. 

<br>

- 이번에는 **효율적인 inference를 위한 몇가지 알고리즘**에 대하여 알아보도록 하겠습니다.

<br>

- 먼저 `pruning` 입니다. Pruning은 불필요하거나 성능에 큰 영향을 주지 않는 파라미터를 줄입니다. 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/cs231n/15/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 pruning은 정확도 하락과 parameter pruning 사이에 trade-off를 가집니다.
- 반복적인 pruning과 retraining을 통하여 Han et al. (NIPS 15)에서는 AlexNet에서 90% 이상의 parameter pruning을 하면서 0.5% 미만의 정확도 하락을 보였습니다.

<br>

- 다음은 `weight sharing`입니다. 이는 비슷한 weight끼리 clustering 하여 같은 숫자로 표현하는 방법입니다.
- 사용하는 parameter들을 적은 수의 숫자로만 표현하므로 정확도는 떨어지는 대신 숫자 표현에 사용되는 bit 수를 대폭 줄일 수 있어 간단하게 만들 수 있습니다. 일종의 `quantization` 방법이라고 말할 수 있습니다.
- `quantization`이란 floating point 타입을 정확도를 크게 손상시키지 않는 선에서 더 적은 bit의 fixed-point로 변경하거나 integer 타입으로 변경하는 것을 말합니다.

<br>
<center><img src="../assets/img/vision/cs231n/15/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>
<center><img src="../assets/img/vision/cs231n/15/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>



<br>

[CS231n 강의 목록](https://gaussian37.github.io/vision-cs231n-table/)

<br>