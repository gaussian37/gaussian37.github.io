---
layout: post
title: Transformer 모델 (Attention is all you need)
date: 2020-11-23 00:00:00
img: dl/concept/transformer/0.png
categories: [dl-concept]
tags: [attention, transformer, attention is all you need] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : fastcampus 딥러닝/인공지능 올인원 패키지 Online.
- 참조 : KoreaUniv DSBA, 08-2: Transformer (Kor) (https://youtu.be/Yk1tV_cXMMU)
- 참조 : https://medium.com/@deepganteam/what-are-transformers-b687f2bcdf49
- 참조 : https://youtu.be/AA621UofTUA?list=RDCMUChflhu32f5EUHlY7_SetNWw

<br>

## **목차**

<br>

- ### Transformer의 특징
- ### Transformer와 Seq2seq의 차이점
- ### Transformer의 입력과 출력
- ### Pytorch 코드
- ### Tensorflow 코드

<br>

## **Transformer의 특징**

<br>
<center><img src="../assets/img/dl/concept/transformer/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- Transformer 모델의 구조는 위 그림과 같습니다. 이 모델은 번역 문제에서 RNN과 CNN을 쓰지 않고 `Attention`과 Fully Connected Layer와 같은 기본 연산만을 이용하여 SOTA 성능을 이끌어낸 연구로 유명합니다.
- 먼저 모델의 아키텍쳐에 대하여 간단히 살펴보겠습니다.
- ① `Seq2seq`와 유사한 Encoder - Decoder 형식을 사용합니다.
- ② Transformer에서 제안하는 `Scaled Dot-Product Attention`과 이를 병렬로 나열한 `Multi-Head Attention` 블록이 알고리즘의 핵심입니다.
- ③ RNN의 구조에서는 아래 그림과 같은 `BPTT` (Back Propagation Through Time) 구조로 인하여 시간 step을 기준을 펼쳐놓고 봐야 하기 때문에 순차적인 연산이 필요합니다. 따라서 연산 효율이 떨어지는 문제가 발생합니다. Transformer 구조에서는 BPTT와 같은 구조가 없고 병렬 계산이 가능하기 때문에 RNN 구조에 비하여 굉장히 효율적으로 연산할 수 있습니다. 

<br>
<center><img src="../assets/img/dl/concept/transformer/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- ④ Transformer 같은 경우 RNN과 다르게 병렬적으로 계산할 수 있기 때문에 현재 계산하고 있는 단어가 어느 위치에 있는 단어인지를 표현을 해주어야 합니다. 그래서 `Positional Encoding`을 사용하고 있습니다.

<br>

## **Transformer와 Seq2seq의 차이점**

<br>
<center><img src="../assets/img/dl/concept/transformer/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 Transformer와 Seq2seq 비교를 통하여 구조를 알아보도록 하겠습니다.
- Seq2seq의 경우 Encoder(빨간색)와 Decoder(파란색) 형태로 이루어져 있고 각 Encoder, Decoder에는 RNN을 사용하였습니다. 그리고 Encoder에서 Decoder로 정보를 전달할 때, 가운데 화살표인 context vector에 Encoder의 모든 정보를 저장하여 Decoder로 전달합니다.
- 반면 Transformer의 경우에도 Encoder(빨간색)와 Decoder(파란색) 형태가 있고 Encoder 끝단 부분에 Decoder로 전달되는 화살표가 있어서 Seq2seq와 유사한 구조를 가집니다. 
- 하지만 **가장 큰 차이점**은 Seq2seq에서는 Encoder에 입력되어야 할 모든 인풋(ex. 단어)인 $$ x_{0}, x_{1}, ... $$이 처리된 후에 Decoder 연산이 시작되나 Transformer에서는 Encoder의 계산과 Decoder의 계산이 동시에 일어나는 차이점이 있습니다.

<br>

## **Transformer의 입력과 출력**

<br>
<center><img src="../assets/img/dl/concept/transformer/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음으로 Transformer의 입력과 출력의 형태에 대하여 알아보도록 하곘습니다.
- Transformer 모델이 자연어 번역을 위해서 개발되었기 때문에 단어 벡터를 입출력 기준으로 설명하겠습니다.
- 전체 입출력 형태를 보면 행렬과 같이 보입니다. 그러면 행렬의 행과 열은 무엇을 의미하는 지 살펴보겠습니다.

<br>

- 먼저 `입력(Input)`에 대하여 살펴보겠습니다.
- 각 단어는 One-hot Encoding 형태의 `벡터`로 나타내어 집니다. 따라서 열벡터가 한 단어에 해당하며 열 벡터의 길이 즉, 행렬에서 행의 크기는 입력 단어의 가짓수와 관련이 있습니다.
- 반면 열의 길이는 사용되는 sequence한 단어의 길이입니다. 문장에서 단어가 10개 있다면 행렬에서의 열의 크기는 10이 된다고 볼 수 있습니다.

<br>

- 다음은 `출력(Output)`에 대하여 살펴보겠습니다.
- Transformer에서 처음 다룬 문제는 번역입니다. 따라서 출력도 단어 형태이나 입력과 출력의 행렬 사이즈는 다를 수 있습니다.
- 예를 들어 입력이 영어고 출력이 한국어이면 사용되는 단어의 갯수도 다를 뿐 아니라 같은 의미의 문장이라도 그 문장에서 사용되는 단어의 갯수도 다를 수 있기 때문입니다.
- 따라서 입력과 출력의 크기는 차이가 있을 수 있으나, 구성 성분은 동일합니다. 출력의 각 열 벡터의 크기 즉, 행렬의 행의 크기는 출력 단어의 가짓수와 관련이 있고 행렬의 열의 크기는 출력에서 사용되는 sequence한 단어의 길이를 나타냅니다.

<br>

- 입력과는 다르게 `출력(Output)`을 살펴보면 2가지의 Output이 표시되어 있습니다.
- Seq2seq에서도 그림에서 위쪽에 있는 Output은 Transformer 모델을 통해 출력되는 실제 출력이고 아래쪽에 있는 Output은 Transformer에서 만들어낸 Output을 다시 입력으로 사용되는 것을 나타냅니다.
- 다시 입력되는 Output의 첫 열벡터는 `SOS (Start Of Sequence)`가 되고 X 표시가 되어있는 마지막 열벡터는 `EOS (End Of Sequence)`이므로 큰 의미는 없는 벡터입니다.
- 따라서 `shifted right`라고 적힌 부분은 Output 에서 하나씩 오른쪽으로 밀려서 다시 입력으로 들어가는 구조로 이해하시면 됩니다.




<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>